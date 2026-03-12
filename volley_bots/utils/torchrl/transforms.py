from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from tensordict.tensordict import TensorDict, TensorDictBase
from tensordict.utils import NestedKey
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    MultiDiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.data.tensor_specs import TensorSpec
from torchrl.envs.common import EnvBase
from torchrl.envs.transforms import (
    CatTensors,
    Compose,
    FlattenObservation,
    Transform,
    TransformedEnv,
)

from .env import AgentSpec


def _transform_agent_spec(self: Transform, agent_spec: AgentSpec) -> AgentSpec:
    return agent_spec


Transform.transform_agent_spec = _transform_agent_spec


def _transform_agent_spec(self: Compose, agent_spec: AgentSpec) -> AgentSpec:
    for transform in self.transforms:
        agent_spec = transform.transform_agent_spec(agent_spec)
    return agent_spec


Compose.transform_agent_spec = _transform_agent_spec


def _agent_spec(self: TransformedEnv) -> AgentSpec:
    agent_spec = self.transform.transform_agent_spec(self.base_env.agent_spec)
    return {name: replace(spec, _env=self) for name, spec in agent_spec.items()}


TransformedEnv.agent_spec = property(_agent_spec)

import logging
import os

import h5py
import numpy as np


def append_to_h5(filename, data_dict: Dict[str, Optional[np.ndarray]]):
    with h5py.File(filename, "a") as file:
        for dataset_name, new_data in data_dict.items():
            if new_data is None:
                continue
            if dataset_name not in file:
                dataset = file.create_dataset(
                    dataset_name, data=new_data, maxshape=(None, new_data.shape[-1])
                )
            else:
                dataset = file[dataset_name]
                # Get the current shape of the dataset
                current_shape = dataset.shape
                # Calculate the new shape after concatenation
                new_shape = (current_shape[0] + new_data.shape[0], current_shape[1])
                # Resize the dataset to accommodate the new data
                dataset.resize(new_shape)
                # Append the new data to the dataset
                dataset[current_shape[0] : new_shape[0], :] = new_data


class _Buffer:
    def __init__(self, size=1024 * 128) -> None:
        self._size = size
        self._cnt = 0
        self._buf = []

    def update(self, arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return

        self._cnt += len(arr)
        self._buf.append(arr)

        if self._cnt >= self._size:
            tmp: np.ndarray = np.concatenate(self._buf, axis=0)
            self._buf.clear()
            self._cnt = 0
            return tmp


class ActionTracker(Transform):
    def __init__(
        self,
        action_key: NestedKey = ("agents", "action"),
        dist_loc_key: NestedKey = ("debug", "action_loc"),
        dist_scale_key: NestedKey = ("debug", "action_scale"),
        maximum_length: int = 1024 * 512,
        filename: str = "action_info.h5",
    ):
        super().__init__(in_keys=[action_key, dist_loc_key, dist_scale_key])
        self.maximum_length = maximum_length
        self._cnt = 0

        self.action_buffer = _Buffer()
        self.loc_buffer = _Buffer()
        self.scale_buffer = _Buffer()

        self.filename = filename

        if os.path.exists(self.filename):
            logging.warning(
                f"{self.filename} already exists. New data will be appended."
            )

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self._cnt >= self.maximum_length:
            return tensordict
        # the original action is on cuda so there is no need to clone.
        action: torch.Tensor = tensordict.get(self.in_keys[0]).cpu()  # (E,A,action_dim)
        action = action.flatten(start_dim=0, end_dim=1).numpy()  # (E*A,action_dim)
        self._cnt += len(action)

        loc: Optional[torch.Tensor] = tensordict.get(self.in_keys[1], None)
        if loc is not None:
            loc: torch.Tensor = loc.cpu()
            loc = loc.flatten(start_dim=0, end_dim=1).numpy()

        scale: Optional[torch.Tensor] = tensordict.get(self.in_keys[2], None)
        if scale is not None:
            scale: torch.Tensor = scale.cpu()
            scale = scale.flatten(start_dim=0, end_dim=1).numpy()
        append_to_h5(
            self.filename,
            {
                "action": self.action_buffer.update(action),
                "loc": self.loc_buffer.update(loc),
                "scale": self.scale_buffer.update(scale),
            },
        )

        return tensordict


class LogOnFirstEpisode(Transform):
    """
    Use in eval.py
    """

    def __init__(
        self,
        in_keys: Sequence[str] = None,
        log_keys: Sequence[str] = None,
        logger_func: Callable = None,
        process_func: Dict[str, Callable] = None,
    ):
        super().__init__(in_keys=in_keys)
        if not len(in_keys) == len(log_keys):
            raise ValueError
        self.in_keys = in_keys
        self.log_keys = log_keys

        self.logger_func = logger_func
        self.process_func = defaultdict(lambda: lambda x: torch.mean(x.float()).item())
        if process_func is not None:
            self.process_func.update(process_func)

        self.stats = []
        self.episode_done: Optional[torch.Tensor] = None

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict

    def _log(self):
        if len(self.stats) == 0:
            return
        stats: TensorDictBase = torch.stack(self.stats)
        dict_to_log = {}
        for in_key, log_key in zip(self.in_keys, self.log_keys):
            try:
                process_func = self.process_func[log_key]
                if isinstance(log_key, tuple):
                    log_key = ".".join(log_key)
                dict_to_log[log_key] = process_func(stats[in_key])
            except:
                pass

                # skip None
        if self.training:
            dict_to_log = {
                f"train/{k}": v for k, v in dict_to_log.items() if v is not None
            }
        else:
            dict_to_log = {
                f"eval/{k}": v for k, v in dict_to_log.items() if v is not None
            }

        if self.logger_func is not None:
            self.logger_func(dict_to_log)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.episode_done is None:
            self.episode_done = torch.zeros(tensordict.batch_size, dtype=torch.bool)
        _reset = tensordict.get(("next", "done"), None)
        if _reset is None:
            _reset = torch.zeros(
                tensordict.batch_size, dtype=torch.bool, device=tensordict.device
            )  # (E,1)

        if _reset.any():
            _reset = _reset.all(-1).cpu()
            index = _reset & ~self.episode_done
            self.episode_done = self.episode_done | _reset
            rst_tensordict = tensordict.get("next").select(*self.in_keys).cpu()
            self.stats.extend(rst_tensordict[index].unbind(0))
            if self.episode_done is not None and self.episode_done.all():
                self._log()
                self.stats.clear()

        return tensordict


class LogOnEpisode(Transform):
    def __init__(
        self,
        n_episodes: int,
        in_keys: Sequence[str] = None,
        log_keys: Sequence[str] = None,
        logger_func: Callable = None,
        process_func: Dict[str, Callable] = None,
    ):
        super().__init__(in_keys=in_keys)
        if not len(in_keys) == len(log_keys):
            raise ValueError
        self.in_keys = in_keys
        self.log_keys = log_keys

        self.n_episodes = n_episodes
        self.logger_func = logger_func

        self.process_func = defaultdict(
            lambda: lambda x: torch.nanmean(x.float()).item()
        )
        if process_func is not None:
            self.process_func.update(process_func)

        self.stats = []
        self._frames = 0

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict

    def _step(self, tensordict: TensorDictBase, next_tensordict) -> TensorDictBase:
        # pdb.set_trace()
        _reset = next_tensordict.get("done", None)
        if _reset is None:
            _reset = torch.zeros(
                tensordict.batch_size, dtype=torch.bool, device=tensordict.device
            )
        if _reset.any():
            # pdb.set_trace()
            _reset = _reset.all(-1).cpu()  # [num_envs,]
            rst_tensordict = next_tensordict.select(*self.in_keys).cpu()
            self.stats.extend(rst_tensordict[_reset].unbind(0))
            if len(self.stats) >= self.n_episodes:
                stats: TensorDictBase = torch.stack(self.stats)
                dict_to_log = {}
                for in_key, log_key in zip(self.in_keys, self.log_keys):
                    try:
                        process_func = self.process_func[log_key]
                        if isinstance(log_key, tuple):
                            log_key = ".".join(log_key)
                        dict_to_log[log_key] = process_func(stats[in_key])
                    except:
                        pass

                # skip None
                if self.training:
                    dict_to_log = {
                        f"train/{k}": v for k, v in dict_to_log.items() if v is not None
                    }
                else:
                    dict_to_log = {
                        f"eval/{k}": v for k, v in dict_to_log.items() if v is not None
                    }

                if self.logger_func is not None:
                    dict_to_log["env_frames"] = self._frames
                    self.logger_func(dict_to_log)
                self.stats.clear()

        if self.training:
            self._frames += tensordict.numel()
        return next_tensordict


class FromDiscreteAction(Transform):
    def __init__(
        self,
        action_key: Tuple[str] = ("agents", "action"),
        nbins: Union[int, Sequence[int]] = None,
    ):
        if nbins is None:
            nbins = 2
        super().__init__([], in_keys_inv=[action_key])
        if not isinstance(action_key, tuple):
            action_key = (action_key,)
        self.nbins = nbins
        self.action_key = action_key

    def transform_input_spec(self, input_spec: CompositeSpec) -> CompositeSpec:
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        if isinstance(action_spec, BoundedTensorSpec):
            if isinstance(self.nbins, int):
                nbins = [self.nbins] * action_spec.shape[-1]
            elif len(self.nbins) == action_spec.shape[-1]:
                nbins = self.nbins
            else:
                raise ValueError(
                    "nbins must be int or list of length equal to the last dimension of action space."
                )
            self.minimum = action_spec.space.minimum.unsqueeze(-2)
            self.maximum = action_spec.space.maximum.unsqueeze(-2)
            self.mapping = torch.cartesian_prod(
                *[torch.linspace(0, 1, dim_nbins) for dim_nbins in nbins]
            ).to(
                action_spec.device
            )  # [prod(nbins), len(nbins)]
            n = self.mapping.shape[0]
            spec = DiscreteTensorSpec(
                n, shape=[*action_spec.shape[:-1], 1], device=action_spec.device
            )
        else:
            NotImplementedError("Only BoundedTensorSpec is supported.")
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec

    def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        mapping = self.mapping * (self.maximum - self.minimum) + self.minimum
        action = action.unsqueeze(-1)
        action = torch.take_along_dim(mapping, action, dim=-2).squeeze(-2)
        return action


class FromMultiDiscreteAction(Transform):
    def __init__(
        self,
        action_key: Tuple[str] = ("agents", "action"),
        nbins: Union[int, Sequence[int]] = 2,
    ):
        if action_key is None:
            action_key = "action"
        super().__init__([], in_keys_inv=[action_key])
        if not isinstance(action_key, tuple):
            action_key = (action_key,)
        self.nbins = nbins
        self.action_key = action_key

    def transform_input_spec(self, input_spec: CompositeSpec) -> CompositeSpec:
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        if isinstance(action_spec, BoundedTensorSpec):
            if isinstance(self.nbins, int):
                nbins = [self.nbins] * action_spec.shape[-1]
            elif len(self.nbins) == action_spec.shape[-1]:
                nbins = self.nbins
            else:
                raise ValueError(
                    "nbins must be int or list of length equal to the last dimension of action space."
                )
            spec = MultiDiscreteTensorSpec(
                nbins, shape=action_spec.shape, device=action_spec.device
            )
            self.nvec = spec.nvec.to(action_spec.device)
            self.minimum = action_spec.space.minimum
            self.maximum = action_spec.space.maximum
        else:
            NotImplementedError("Only BoundedTensorSpec is supported.")
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec

    def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        action = action / (self.nvec - 1) * (self.maximum - self.minimum) + self.minimum
        return action

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return super()._inv_call(tensordict)


class DepthImageNorm(Transform):
    def __init__(
        self,
        in_keys: Sequence[str],
        min_range: float,
        max_range: float,
        inverse: bool = False,
    ):
        super().__init__(in_keys=in_keys)
        self.max_range = max_range
        self.min_range = min_range
        self.inverse = inverse

    def _apply_transform(self, obs: torch.Tensor) -> None:
        obs = torch.nan_to_num(obs, posinf=self.max_range, neginf=self.min_range)
        obs = obs.clip(self.min_range, self.max_range)
        if self.inverse:
            obs = (obs - self.min_range) / (self.max_range - self.min_range)
        else:
            obs = (self.max_range - obs) / (self.max_range - self.min_range)
        return obs


def ravel_composite(
    spec: CompositeSpec, key: str, start_dim: int = -2, end_dim: int = -1
):
    r"""

    Examples:
    >>> obs_spec = CompositeSpec({
    ...     "obs_self": UnboundedContinuousTensorSpec((1, 19)),
    ...     "obs_others": UnboundedContinuousTensorSpec((3, 13)),
    ... })
    >>> spec = CompositeSpec({
            "agents": {
                "observation": obs_spec
            }
    ... })
    >>> t = ravel_composite(spec, ("agents", "observation"))

    """
    composite_spec = spec[key]
    if not isinstance(key, tuple):
        key = (key,)
    if isinstance(composite_spec, CompositeSpec):
        in_keys = [k for k in spec.keys(True, True) if k[: len(key)] == key]
        return Compose(
            FlattenObservation(start_dim, end_dim, in_keys),
            CatTensors(in_keys, out_key=key),
        )
    else:
        raise TypeError


class PosController(Transform):
    def __init__(
        self,
        controller,
        action_key: str = ("agents", "action"),
    ):
        super().__init__([], in_keys_inv=[("info", "drone_state")])
        self.controller = controller
        self.action_key = action_key

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        spec = UnboundedContinuousTensorSpec(
            action_spec.shape[:-1] + (7,), device=action_spec.device
        )
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        drone_state = tensordict[("info", "drone_state")][..., :13]
        action = tensordict[self.action_key]
        target_pos, target_vel, target_yaw = action.split([3, 3, 1], -1)
        cmds = self.controller(
            drone_state,
            target_pos=target_pos
            - drone_state[..., :3],  # using relative position to learn
            target_vel=target_vel,
            target_yaw=target_yaw * torch.pi,
        )
        torch.nan_to_num_(cmds, 0.0)
        tensordict.set(self.action_key, cmds)
        return tensordict


class VelController(Transform):
    def __init__(
        self,
        controller,
        action_key: str = ("agents", "action"),
    ):
        super().__init__([], in_keys_inv=[("info", "drone_state")])
        self.controller = controller
        self.action_key = action_key

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        spec = UnboundedContinuousTensorSpec(
            action_spec.shape[:-1] + (4,), device=action_spec.device
        )
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        drone_state = tensordict[("info", "drone_state")][..., :13]
        action = tensordict[self.action_key]
        target_vel, target_yaw = action.split([3, 1], -1)
        cmds = self.controller(
            drone_state, target_vel=target_vel, target_yaw=target_yaw * torch.pi
        )
        torch.nan_to_num_(cmds, 0.0)
        tensordict.set(self.action_key, cmds)
        return tensordict


class RateController(Transform):
    def __init__(
        self,
        controller,
        action_key: str = ("agents", "action"),
    ):
        super().__init__([], in_keys_inv=[("info", "drone_state")])
        self.controller = controller
        self.action_key = action_key
        self.max_thrust = self.controller.max_thrusts.sum(-1)

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        spec = UnboundedContinuousTensorSpec(
            action_spec.shape[:-1] + (4,), device=action_spec.device
        )
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        drone_state = tensordict[("info", "drone_state")][..., :13]
        action = tensordict[self.action_key]
        target_rate, target_thrust = action.split([3, 1], -1)
        target_thrust = ((target_thrust + 1) / 2).clip(0.0) * self.max_thrust
        cmds = self.controller(
            drone_state, target_rate=target_rate * torch.pi, target_thrust=target_thrust
        )
        torch.nan_to_num_(cmds, 0.0)
        tensordict.set(self.action_key, cmds)
        return tensordict


class PIDRateController(Transform):
    def __init__(
        self,
        controller,
        action_key: str = ("agents", "action"),
    ):
        super().__init__([], in_keys_inv=[("info", "drone_state")])
        self.controller = controller
        self.action_key = action_key
        self.max_thrust = self.controller.max_thrusts.sum(-1)
        self.target_clip = self.controller.target_clip
        self.max_thrust_ratio = self.controller.max_thrust_ratio
        self.fixed_yaw = self.controller.fixed_yaw
        # for action smooth
        self.use_action_smooth = self.controller.use_action_smooth
        self.use_cbf = self.controller.use_cbf
        self.epsilon = self.controller.epsilon
        # self.tanh = TanhTransform()

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        spec = UnboundedContinuousTensorSpec(
            action_spec.shape[:-1] + (4,), device=action_spec.device
        )
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        drone_state = tensordict[("info", "drone_state")][..., :13]
        action = tensordict[self.action_key]
        device = drone_state.device

        # if not(self.epsilon is None) and self.use_cbf:
        #     action = solve_qp_batch(action.to('cpu').numpy(), prev_action.to('cpu').numpy(), self.epsilon)
        #     action = torch.from_numpy(action).to(device).float()

        # target CTBR:
        # target_rate: [-1, 1]
        # target_thrust: [0, max_thrust_ratio]
        action = torch.tanh(action)
        target_rate, target_thrust = action.split([3, 1], -1)
        target_thrust = torch.clamp(
            (target_thrust + 1) / 2, min=0.0, max=self.max_thrust_ratio
        )
        if self.fixed_yaw:
            target_rate[..., 2] = 0.0

        # raw action error
        ctbr_action = torch.concat([target_rate, target_thrust], dim=-1)

        prev_ctbr_action = tensordict[("info", "prev_action")]
        prev_prev_ctbr_action = tensordict[("info", "prev_prev_action")]

        # action smoothness
        if not (self.epsilon is None) and self.use_action_smooth:
            ctbr_action = prev_ctbr_action + torch.clamp(
                ctbr_action - prev_ctbr_action, min=-self.epsilon, max=+self.epsilon
            )
            target_rate, target_thrust = ctbr_action.split([3, 1], -1)

        action_error = torch.norm(ctbr_action - prev_ctbr_action, dim=-1)
        tensordict.set(("stats", "action_error_order1"), action_error)
        action_error_2 = torch.norm(
            prev_prev_ctbr_action + ctbr_action - 2 * prev_ctbr_action, dim=-1
        )
        tensordict.set(("stats", "action_error_order2"), action_error_2)
        # update prev_action = current ctbr_action
        tensordict.set(("info", "prev_action"), ctbr_action)
        # update prev_prev_action =  prev_ctbr_action
        tensordict.set(("info", "prev_prev_action"), prev_ctbr_action)

        # scale
        # target_rate: [-180, 180] degree/s
        # target_thrust: [0, 2^16]
        target_rate = target_rate * 180.0 * self.target_clip
        target_thrust = target_thrust * 2**16

        # current rotors cmds and CTBR
        cmds, ctbr = self.controller(
            drone_state,
            target_rate=target_rate,
            target_thrust=target_thrust,
            reset_pid=tensordict["done"].expand(
                -1, drone_state.shape[1]
            ),  # num_drones: drone_state.shape[1]
        )
        torch.nan_to_num_(cmds, 0.0)
        tensordict.set(self.action_key, cmds)
        tensordict.set("ctbr", ctbr)
        tensordict.set("target_rate", target_rate)
        tensordict.set("target_thrust", target_thrust)
        # import pdb; pdb.set_trace()
        return tensordict


class AttitudeController(Transform):
    def __init__(
        self,
        controller,
        action_key: str = ("agents", "action"),
    ):
        super().__init__([], in_keys_inv=[("info", "drone_state")])
        self.controller = controller
        self.action_key = action_key
        self.max_thrust = self.controller.max_thrusts.sum(-1)

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        spec = UnboundedContinuousTensorSpec(
            action_spec.shape[:-1] + (4,), device=action_spec.device
        )
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        drone_state = tensordict[("info", "drone_state")][..., :13]
        action = tensordict[self.action_key]
        target_thrust, target_yaw_rate, target_roll, target_pitch = action.split(
            1, dim=-1
        )
        cmds = self.controller(
            drone_state,
            target_thrust=((target_thrust + 1) / 2).clip(0.0) * self.max_thrust,
            target_yaw_rate=target_yaw_rate * torch.pi,
            target_roll=target_roll * torch.pi,
            target_pitch=target_pitch * torch.pi,
        )
        torch.nan_to_num_(cmds, 0.0)
        tensordict.set(self.action_key, cmds)
        return tensordict


class CosmosCommandTransform(Transform):
    """Appends a cached high-level command (one-hot) to the observation vector.

    When a :class:`~volley_bots.utils.cosmos.CosmosReasoner` and ``base_env``
    are provided, the transform periodically renders frames from the
    environment, feeds them to the VLM, and uses the resulting command.
    Otherwise it falls back to random commands during training.

    At eval time the caller can set specific commands via :meth:`set_command`.
    """

    def __init__(
        self,
        num_commands: int = 6,
        num_envs: int = 1,
        call_every_k: int = 50,
        device: str = "cuda",
        reasoner=None,
        base_env=None,
        record_every: int = 5,
        max_frames: int = 8,
        logger_func: Callable[[Dict[str, Any]], None] = None,
        counterfactual_enabled: bool = True,
        counterfactual_threshold: float = 0.1,
    ):
        super().__init__(in_keys=[("agents", "observation")])
        self.num_commands = num_commands
        self.call_every_k = call_every_k
        self._step_count = 0
        self.cached_cmd = torch.zeros(num_envs, 1, num_commands, device=device)
        self.cached_cmd[..., 0] = 1.0  # default: first command

        self.reasoner = reasoner
        self.base_env = base_env
        self.record_every = record_every
        self.max_frames = max_frames
        self._frame_buffer: list = []
        self.last_reasoning: str = ""
        self.last_command_name: str = ""
        self.logger_func = logger_func
        self.counterfactual_enabled = counterfactual_enabled
        self.counterfactual_threshold = counterfactual_threshold
        self.policy = None
        self._cf_call_count = 0
        self._cf_running_stats = defaultdict(
            lambda: {
                "calls": 0,
                "action_l2": 0.0,
                "action_l1": 0.0,
                "normalized_l2": 0.0,
                "cosine_distance": 0.0,
                "changed_fraction": 0.0,
                "sign_flip_fraction": 0.0,
                "value_delta_abs": 0.0,
            }
        )
        self._cf_running_by_chosen = defaultdict(
            lambda: {
                "calls": 0,
                "mean_alt_action_l2": 0.0,
                "mean_alt_normalized_l2": 0.0,
                "mean_alt_value_delta_abs": 0.0,
            }
        )

        import logging

        self._logger = logging.getLogger(__name__)

    @property
    def use_reasoner(self) -> bool:
        return self.reasoner is not None and self.base_env is not None

    def transform_observation_spec(self, observation_spec):
        spec = observation_spec[("agents", "observation")]
        new_shape = (*spec.shape[:-1], spec.shape[-1] + self.num_commands)
        observation_spec[("agents", "observation")] = UnboundedContinuousTensorSpec(
            new_shape, device=spec.device
        )
        return observation_spec

    def attach_policy(
        self,
        policy,
        logger_func: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """Attach the rollout policy for side-effect-only counterfactual probes."""
        self.policy = policy
        if logger_func is not None:
            self.logger_func = logger_func

    def _append_command(
        self, tensordict: TensorDictBase, command_tensor: torch.Tensor
    ) -> TensorDictBase:
        obs = tensordict.get(("agents", "observation"))
        cmd = command_tensor.expand(*obs.shape[:-1], self.num_commands)
        tensordict.set(("agents", "observation"), torch.cat([obs, cmd], dim=-1))
        return tensordict

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self._append_command(tensordict, self.cached_cmd)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    def _capture_frame(self):
        """Render and buffer a frame from the environment."""
        frame = self.base_env.render(mode="rgb_array")
        self._frame_buffer.append(frame)

    def _run_policy_probe(self, tensordict: TensorDictBase):
        if self.policy is None:
            return None
        probe_td = tensordict.clone()
        with torch.no_grad():
            try:
                return self.policy(probe_td, deterministic=True)
            except TypeError:
                return self.policy(probe_td)

    @staticmethod
    def _extract_scalar(value: Optional[torch.Tensor]) -> float:
        if value is None:
            return 0.0
        if not isinstance(value, torch.Tensor):
            return float(value)
        return value.detach().float().mean().item()

    def _log_counterfactual_metrics(
        self,
        chosen_idx: int,
        base_output: TensorDictBase,
        raw_tensordict: TensorDictBase,
    ):
        if (
            not self.counterfactual_enabled
            or self.policy is None
            or self.logger_func is None
        ):
            return

        base_action = base_output.get(("agents", "action"), None)
        if base_action is None:
            return

        base_action = base_action.detach()
        base_value = base_output.get("state_value", None)
        base_value = None if base_value is None else base_value.detach()
        base_norm = base_action.norm(dim=-1).mean()
        command_names = getattr(self.reasoner, "commands", None) or [
            f"command_{i}" for i in range(self.num_commands)
        ]
        prefix = "train" if self.training else "eval"

        table = None
        try:
            import wandb

            table = wandb.Table(
                columns=[
                    "step",
                    "chosen_command",
                    "candidate_command",
                    "is_chosen",
                    "action_l2",
                    "action_l1",
                    "normalized_l2",
                    "cosine_distance",
                    "changed_fraction",
                    "sign_flip_fraction",
                    "value_delta_abs",
                ]
            )
        except Exception:
            table = None

        call_metrics = {}
        alt_l2_values = []
        alt_norm_l2_values = []
        alt_value_delta_values = []

        for cmd_idx, cmd_name in enumerate(command_names):
            cmd_tensor = torch.zeros_like(self.cached_cmd)
            cmd_tensor[..., cmd_idx] = 1.0
            probe_td = raw_tensordict.clone()
            probe_td = self._append_command(probe_td, cmd_tensor)
            probe_output = self._run_policy_probe(probe_td)
            if probe_output is None:
                return

            probe_action = probe_output.get(("agents", "action"), None)
            if probe_action is None:
                return
            probe_action = probe_action.detach()
            delta = probe_action - base_action

            action_l2 = delta.norm(dim=-1).mean()
            action_l1 = delta.abs().mean()
            normalized_l2 = action_l2 / (base_norm + 1e-8)
            cosine_distance = (
                1.0
                - F.cosine_similarity(base_action, probe_action, dim=-1, eps=1e-8).mean()
            )
            changed_fraction = (
                (delta.abs() > self.counterfactual_threshold).float().mean()
            )
            sign_flip_fraction = ((base_action * probe_action) < 0).float().mean()

            probe_value = probe_output.get("state_value", None)
            if base_value is not None and probe_value is not None:
                value_delta_abs = (probe_value.detach() - base_value).abs().mean()
            else:
                value_delta_abs = None

            metrics = {
                "action_l2": self._extract_scalar(action_l2),
                "action_l1": self._extract_scalar(action_l1),
                "normalized_l2": self._extract_scalar(normalized_l2),
                "cosine_distance": self._extract_scalar(cosine_distance),
                "changed_fraction": self._extract_scalar(changed_fraction),
                "sign_flip_fraction": self._extract_scalar(sign_flip_fraction),
                "value_delta_abs": self._extract_scalar(value_delta_abs),
            }

            if table is not None:
                table.add_data(
                    int(self._step_count),
                    command_names[chosen_idx],
                    cmd_name,
                    int(cmd_idx == chosen_idx),
                    metrics["action_l2"],
                    metrics["action_l1"],
                    metrics["normalized_l2"],
                    metrics["cosine_distance"],
                    metrics["changed_fraction"],
                    metrics["sign_flip_fraction"],
                    metrics["value_delta_abs"],
                )

            if cmd_idx != chosen_idx:
                alt_l2_values.append(metrics["action_l2"])
                alt_norm_l2_values.append(metrics["normalized_l2"])
                alt_value_delta_values.append(metrics["value_delta_abs"])

                for metric_name, metric_value in metrics.items():
                    call_metrics[
                        f"{prefix}/cosmos_counterfactual/by_command/{cmd_name}/{metric_name}"
                    ] = metric_value

                running_stats = self._cf_running_stats[cmd_name]
                running_stats["calls"] += 1
                for metric_name, metric_value in metrics.items():
                    running_stats[metric_name] += metric_value

        self._cf_call_count += 1
        chosen_name = command_names[chosen_idx]
        chosen_stats = self._cf_running_by_chosen[chosen_name]
        chosen_stats["calls"] += 1
        chosen_stats["mean_alt_action_l2"] += float(np.mean(alt_l2_values))
        chosen_stats["mean_alt_normalized_l2"] += float(np.mean(alt_norm_l2_values))
        chosen_stats["mean_alt_value_delta_abs"] += float(np.mean(alt_value_delta_values))

        summary = {
            f"{prefix}/cosmos_counterfactual/step": int(self._step_count),
            f"{prefix}/cosmos_counterfactual/calls": self._cf_call_count,
            f"{prefix}/cosmos_counterfactual/chosen_command_idx": chosen_idx,
            f"{prefix}/cosmos_counterfactual/chosen_command": chosen_name,
            f"{prefix}/cosmos_counterfactual/summary/mean_alt_action_l2": float(
                np.mean(alt_l2_values)
            ),
            f"{prefix}/cosmos_counterfactual/summary/max_alt_action_l2": float(
                np.max(alt_l2_values)
            ),
            f"{prefix}/cosmos_counterfactual/summary/min_alt_action_l2": float(
                np.min(alt_l2_values)
            ),
            f"{prefix}/cosmos_counterfactual/summary/mean_alt_normalized_l2": float(
                np.mean(alt_norm_l2_values)
            ),
            f"{prefix}/cosmos_counterfactual/summary/mean_alt_value_delta_abs": float(
                np.mean(alt_value_delta_values)
            ),
            f"{prefix}/cosmos_counterfactual/summary/base_action_norm": self._extract_scalar(
                base_norm
            ),
        }
        summary.update(call_metrics)

        for cmd_name, running_stats in self._cf_running_stats.items():
            calls = max(1, running_stats["calls"])
            for metric_name, metric_sum in running_stats.items():
                if metric_name == "calls":
                    continue
                summary[
                    f"{prefix}/cosmos_counterfactual_running/by_command/{cmd_name}/{metric_name}"
                ] = metric_sum / calls

        for chosen_cmd, running_stats in self._cf_running_by_chosen.items():
            calls = max(1, running_stats["calls"])
            summary[
                f"{prefix}/cosmos_counterfactual_running/by_chosen/{chosen_cmd}/calls"
            ] = running_stats["calls"]
            summary[
                f"{prefix}/cosmos_counterfactual_running/by_chosen/{chosen_cmd}/mean_alt_action_l2"
            ] = running_stats["mean_alt_action_l2"] / calls
            summary[
                f"{prefix}/cosmos_counterfactual_running/by_chosen/{chosen_cmd}/mean_alt_normalized_l2"
            ] = running_stats["mean_alt_normalized_l2"] / calls
            summary[
                f"{prefix}/cosmos_counterfactual_running/by_chosen/{chosen_cmd}/mean_alt_value_delta_abs"
            ] = running_stats["mean_alt_value_delta_abs"] / calls

        if table is not None:
            summary[f"{prefix}/cosmos_counterfactual/table"] = table

        self.logger_func(summary)

    def _update_command_from_reasoner(self, next_tensordict: TensorDictBase):
        """Query the Cosmos reasoner with buffered frames."""
        if not self._frame_buffer:
            raise RuntimeError("No frames buffered for Cosmos inference")

        cmd_idx, reasoning, raw = self.reasoner.get_command_with_reasoning(
            self._frame_buffer, max_frames=self.max_frames
        )
        self._frame_buffer.clear()

        n = self.cached_cmd.shape[0]
        cmd_indices = torch.full(
            (n,), cmd_idx, device=self.cached_cmd.device, dtype=torch.long
        )
        self.set_command(cmd_indices)

        self.last_command_name = self.reasoner.commands[cmd_idx]
        self.last_reasoning = reasoning

        base_probe_td = next_tensordict.clone()
        base_probe_td = self._append_command(base_probe_td, self.cached_cmd)
        base_output = self._run_policy_probe(base_probe_td)
        if base_output is not None:
            self._log_counterfactual_metrics(cmd_idx, base_output, next_tensordict)

        if not reasoning.strip():
            reasoning = "[NO_REASONING_RETURNED]"

        print(
            f"\n{'='*60}\n"
            f"[Cosmos] Step {self._step_count} -> command: {self.last_command_name}\n"
            f"{'-'*60}\n"
            f"Reasoning:\n{reasoning}\n"
            f"{'='*60}\n",
            flush=True,
        )

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        self._step_count += 1

        if self.use_reasoner and self._step_count % self.record_every == 0:
            self._capture_frame()

        if self._step_count % self.call_every_k == 0 and self.use_reasoner:
            self._update_command_from_reasoner(next_tensordict)

        return self._call(next_tensordict)

    # ------------------------------------------------------------------
    # Public helpers for setting commands from outside
    # ------------------------------------------------------------------

    def set_command(self, cmd_indices: torch.Tensor):
        """Set per-env commands from index tensor of shape ``(num_envs,)``."""
        self.cached_cmd.zero_()
        onehot = torch.nn.functional.one_hot(
            cmd_indices.long(), self.num_commands
        ).float()
        self.cached_cmd[:, 0, :] = onehot


class History(Transform):
    def __init__(
        self,
        in_keys: Sequence[str],
        out_keys: Sequence[str] = None,
        steps: int = 32,
    ):
        if out_keys is None:
            out_keys = [
                f"{key}_h" if isinstance(key, str) else key[:-1] + (f"{key[-1]}_h",)
                for key in in_keys
            ]
        if any(key in in_keys for key in out_keys):
            raise ValueError
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.steps = steps

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            is_tuple = isinstance(in_key, tuple)
            if in_key in observation_spec.keys(include_nested=is_tuple):
                spec = observation_spec[in_key]
                spec = spec.unsqueeze(-1).expand(*spec.shape, self.steps)
                observation_spec[out_key] = spec
        return observation_spec

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            item = tensordict.get(in_key)
            item_history = tensordict.get(out_key)
            item_history[..., :-1] = item_history[..., 1:]
            item_history[..., -1] = item
        return tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            item = tensordict.get(in_key)
            item_history = tensordict.get(out_key).clone()
            item_history[..., :-1] = item_history[..., 1:]
            item_history[..., -1] = item
            tensordict.set(("next", out_key), item_history)
        return tensordict

    def reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        _reset = tensordict.get("_reset", None)
        if _reset is None:
            _reset = torch.ones(
                tensordict.batch_size, dtype=bool, device=tensordict.device
            )
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            if out_key not in tensordict.keys(True, True):
                item = tensordict.get(in_key)
                item_history = (
                    item.unsqueeze(-1).expand(*item.shape, self.steps).clone().zero_()
                )
                tensordict.set(out_key, item_history)
            else:
                item_history = tensordict.get(out_key)
                item_history[_reset] = 0.0
        return tensordict

import os
import json
import io
import base64
import requests
import torch
import torch.nn as nn
from tensordict import TensorDict
from volley_bots.utils.torchrl import AgentSpec
from volley_bots.learning import MAPPOPolicy

# reasoning + policy layer

class CosmosPolicy(nn.Module):

    def __init__(self,
        cfg,
        agent_spec: AgentSpec,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.agent_spec = agent_spec
        self.device = device

        self.action_dim = agent_spec.action_spec.shape[-1]

        # Match TD3's naming convention
        if cfg.get("agent_name"):
            self.obs_name = ("agents", cfg.agent_name + "_observation")
            self.act_name = ("agents", cfg.agent_name + "_action")
            self.reward_name = ("agents", cfg.agent_name + "_reward")
        else:
            self.obs_name = ("agents", "observation")
            self.act_name = ("agents", "action")
            self.reward_name = ("agents", "reward")

        # TODO: API CONFIGURATIONS

        self.api_url = cfg.get("llm_api_url", "https://router.huggingface.co/v1/chat/completions")
        self.api_key = cfg.get("llm_api_key", os.environ.get("LLAMA_INSTRUCT_API_KEY", ""))
        self.model   = cfg.get("llm_model", "meta-llama/Llama-3.1-8B-Instruct")
        self.llm_interval = cfg.get("llm_interval", 50) # how often to call LLM
        self.use_vlm = cfg.get("use_vlm", False)
        self.vlm_max_frames = int(cfg.get("vlm_max_frames", 4))
        self.steps_since_llm = 0
        self.current_reasoning = "Initializing..."
        self.cached_video_frames = []
        rl_policy = MAPPOPolicy(cfg, agent_spec, device)
        rl_ckpt = cfg.get("rl_checkpoint_path", None)
        if rl_ckpt: # LOAD SOME ALREADY TRAINED MODEL
            rl_policy.load_state_dict(torch.load(rl_ckpt))
            print(f"[CosmosPolicy] Loaded RL policy from {rl_ckpt}")
        else:
            print("[CosmosPolicy] WARNING: no rl_checkpoint_path, RL policy is untrained")
        object.__setattr__(self, 'rl_policy', rl_policy)
        self.reasoning_log = []

        # action_list = {} # some list of predetermined actions

        # ACTION
        # self.system_prompt = (
        #     f"You are coaching a drone to play volleyball."
        #     f"Given the current game state, output a JSON object with two fields:\n"
        #     f"Goal: Pick a valid action from the list of actions: {action_list}"
        #     f"Reasoning: 1-2 sentences explaining why\n"
        #     f"Output valid JSON and nothing else."
        # )

        # OBSERVATION
        self.system_prompt = (
            f"You are coaching a drone to play volleyball."
            f"Given the current game state, output an explanation of what is going on, what each drone should"
            f"look out for, and strategy for the rest of the episode. In two sentences."
        )

    def _game_state_summary(self, obs: torch.Tensor) -> str:
        # create observation vector from obs tensor. TODO: make these more informative
        # TODO: I literally don't know what the values represent
        o = obs.cpu().tolist()
        return (f"Observation vector {len(o)} values:\n"
                + "\n".join(f"  [{i}]: {v:.3f}" for i, v in enumerate(o))) # figure out what the indices map to lol

    def _query_llm(self, obs: torch.Tensor):
        print(f"Obs: {obs}")
        state_summary = self._game_state_summary(obs)
        user_content = f"Current game state:\n{state_summary}"
        payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": user_content},
                ],
                "max_tokens": 300, # can edit these # this is lowkey not high enough we have to bump it up
                "temperature": 0.2, # can edit these
            }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=10)
            print(f"[CosmosPolicy] Status: {response.status_code}")
            # print(f"[CosmosPolicy] Raw response: {response.text}")

            response.raise_for_status()
            message_str = response.json()["choices"][0]["message"]["content"]
            return message_str
        except Exception as e:
            print(f"[CosmosPolicy] API call failed: {e}")
            return self.current_reasoning
        # print(f"LLM output: message_str")

    def _encode_frame_as_data_url(self, frame):
        """Helper function to encode a video frame as a base64 data URL for VLM input."""
        try:
            from PIL import Image

            image = Image.fromarray(frame)
            with io.BytesIO() as buffer:
                image.save(buffer, format="JPEG")
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}"
        except Exception as exc:
            print(f"[CosmosPolicy] Failed to encode frame for VLM: {exc}")
            return None

    def set_video_frames(self, frames):
        if frames is None:
            self.cached_video_frames = []
            return
        self.cached_video_frames = [frame.copy() for frame in frames]
    
    # for cosmos or whatever VLA we end up usiing
    def _query_vlm(self, obs: torch.Tensor, video_frames=None):
        print(f"Obs: {obs}")
        # Convert structured observation tensor into prompt-friendly state text.
        state_summary = self._game_state_summary(obs)
        video_frames = video_frames if video_frames is not None else []

        if len(video_frames) > 0:
            # Uniformly sample frames to bound payload size and latency.
            sample_count = min(self.vlm_max_frames, len(video_frames))
            sample_indices = torch.linspace(
                0, len(video_frames) - 1, steps=sample_count
            ).long().tolist()
            sampled_frames = [video_frames[i] for i in sample_indices]
        else:
            sampled_frames = []

        # First user message part is textual trajectory context.
        user_content = [
            {
                "type": "text",
                "text": (
                    "Current game state and trajectory context:\n"
                    f"{state_summary}\n"
                    f"Captured evaluation frames: {len(video_frames)}"
                ),
            }
        ]

        for frame in sampled_frames:
            # Attach sampled frames as inline images for multimodal reasoning.
            data_url = self._encode_frame_as_data_url(frame)
            if data_url is not None:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    }
                )

        payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": user_content},
                ],
                "max_tokens": 150, # can edit these
                "temperature": 0.2, # can edit these
            }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(self.api_url, headers=headers, json=payload, timeout=10)
        message = response.json()["choices"][0]["message"]
        message_str = message.get("content", message)
        # print(f"LLM output: message_str")
        return message_str

    @torch.no_grad()
    def forward(self, tensordict: TensorDict) -> TensorDict:
        if self.steps_since_llm >= self.llm_interval:
            obs = tensordict[self.obs_name][0, 0]  # first env, first agent
            if self.use_vlm:
                self.current_reasoning = self._query_vlm(obs, self.cached_video_frames)
            else:
                self.current_reasoning = self._query_llm(obs)
            print(f"[Agent] Reasoning: {self.current_reasoning}\n")

            self.reasoning_log.append({
                "step":      self.steps_since_llm,
                "reasoning": self.current_reasoning,
            })
            self.steps_since_llm = 0

        self.steps_since_llm += 1

        # RL policy does the actual fwding
        rl = object.__getattribute__(self, 'rl_policy')
        return rl(tensordict)

    def train_op(self, data: TensorDict) -> dict:
        return self.rl_policy.train_op(data)

    def state_dict(self):
        rl = object.__getattribute__(self, 'rl_policy')
        return rl.state_dict() if hasattr(rl, 'state_dict') else {}

    def load_state_dict(self, state_dict):
        rl = object.__getattribute__(self, 'rl_policy')
        if hasattr(rl, 'load_state_dict'):
            rl.load_state_dict(state_dict)

    def save_reasoning_log(self, path="reasoning_log.json"):
        with open(path, "w") as f:
            json.dump(self.reasoning_log, f, indent=2)
        print(f"[CosmosPolicy] Reasoning log saved to {path}")
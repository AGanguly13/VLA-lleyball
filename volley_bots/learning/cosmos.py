import os
import json
import requests
import torch
import torch.nn as nn
from tensordict import TensorDict
from volley_bots.utils.torchrl import AgentSpec


class CosmosPolicy(object):

    def __init__(self,
        cfg,
        agent_spec: AgentSpec,
        device: str = "cuda",
    ) -> None:
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
        action_list = {} # some list of predetermined actions

        self.system_prompt = (
            f"You are coaching a drone to play volleyball."
            f"Given the current game state, output a JSON object with two fields:\n"
            f"Goal: Pick a valid action from the list of actions: {action_list}"
            f"Reasoning: 1-2 sentences explaining why\n"
            f"Output valid JSON and nothing else."
        )
"""
TODO: I have many open questions.
- Is NVIDIA Cosmos the right model?
- How will Cosmos know how to interpret random numbers that mean game state? *****
- How will Cosmos know to return continuous values that make the drone perform as expected?
(I think it won't so we should maybe train our RL policy to take in an action from the LLM to perform)
- Is there some way to actually integrate a VLA who understands IsaacSim into the sim environment itself?
- Will the VLA be able to make decisions in time to actually execute them? The API calls are going to take longer than the sim.
"""
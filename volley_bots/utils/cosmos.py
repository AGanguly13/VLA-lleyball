import base64
import io
import json
import logging
import urllib.request
from typing import List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

COSMOS_COMMANDS = [
    "hover",
    "move_to_ball",
    "receive",
    "set",
    "attack",
    "retreat",
]


class CosmosReasoner:
    """Wrapper around NVIDIA Cosmos Reason2 for high-level volleyball strategy.

    Calls an OpenAI-compatible API (e.g. vLLM) over HTTP.  Raises on
    failure — the server must be running and reachable.
    """

    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Reason2-8B",
        commands: Optional[List[str]] = None,
        max_new_tokens: int = 1024,
        server_url: str = "http://localhost:8000",
        timeout: float = 60.0,
    ):
        self.commands = commands or COSMOS_COMMANDS
        self.num_commands = len(self.commands)
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

        req = urllib.request.Request(f"{self.server_url}/v1/models")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        models = [m["id"] for m in data.get("data", [])]
        logger.info(
            f"Connected to inference server at {self.server_url} — "
            f"available models: {models}"
        )

    # ------------------------------------------------------------------
    # Frame encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _frame_to_b64(frame) -> str:
        """Convert a single RGB frame (numpy/tensor) to a base64 JPEG string."""
        from PIL import Image

        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        img = Image.fromarray(frame.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _subsample(frames: List, max_frames: int) -> List:
        if len(frames) > max_frames:
            indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
            return [frames[i] for i in indices]
        return frames

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _build_prompt(self, state_text: Optional[str] = None) -> str:
        command_list = ", ".join(f"'{c}'" for c in self.commands)
        prompt = (
            "You are watching a sequence of frames from a drone volleyball simulation. "
            "Analyze the current game state and choose exactly ONE high-level command "
            f"for the drone to execute from: [{command_list}].\n\n"
            "Think step by step:\n"
            "1. What is the current position of the drone relative to the court?\n"
            "2. Where is the ball and how is it moving?\n"
            "3. What phase of play is this (serving, receiving, setting, attacking)?\n"
            "4. What should the drone do next?\n"
        )
        if state_text:
            prompt += f"\nCurrent state: {state_text}\n"
        prompt += (
            "\n<think>\nYour reasoning.\n</think>\n\n"
            "Write ONLY the chosen action name after </think>."
        )
        return prompt

    def _build_messages(self, frames: List, state_text: Optional[str], max_frames: int):
        """Build OpenAI-format messages with image content.

        Includes a partial assistant message starting with ``<think>`` so the
        model is forced to produce chain-of-thought reasoning before the
        command.
        """
        frames = self._subsample(frames, max_frames)
        image_content = []
        for frame in frames:
            b64 = self._frame_to_b64(frame)
            image_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )

        prompt = self._build_prompt(state_text)
        return [
            {
                "role": "system",
                "content": (
                    "You are an expert volleyball strategist controlling "
                    "a drone in a physics simulation."
                ),
            },
            {
                "role": "user",
                "content": image_content + [{"type": "text", "text": prompt}],
            },
            {
                "role": "assistant",
                "content": "<think>\n",
            },
        ]

    # ------------------------------------------------------------------
    # Inference via vLLM / OpenAI-compatible API
    # ------------------------------------------------------------------

    def get_command_with_reasoning(
        self,
        frames: List,
        state_text: Optional[str] = None,
        max_frames: int = 8,
    ) -> Tuple[int, str, str]:
        """Get a high-level command and chain-of-thought from Cosmos.

        Args:
            frames: list of RGB numpy arrays (H, W, 3).
            state_text: optional natural-language summary of game state.
            max_frames: subsample to at most this many frames.

        Returns:
            ``(command_index, reasoning_text, raw_output)`` tuple.
            *reasoning_text* is the content inside ``<think>…</think>`` tags.

        Raises:
            On any server or inference error.
        """
        messages = self._build_messages(frames, state_text, max_frames)

        payload = json.dumps({
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": 0.7,
        }).encode()

        req = urllib.request.Request(
            f"{self.server_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read())

        output_text = result["choices"][0]["message"]["content"]
        # The assistant message was primed with "<think>\n", so prepend it
        # back to reconstruct the full response.
        full_text = "<think>\n" + output_text
        reasoning = self._extract_reasoning(full_text)
        cmd_idx = self._parse_command(full_text)
        return cmd_idx, reasoning, full_text

    def get_command(
        self, frames: List, state_text: Optional[str] = None
    ) -> int:
        """Get a high-level command index from Cosmos given rendered frames."""
        cmd_idx, _, _ = self.get_command_with_reasoning(frames, state_text)
        return cmd_idx

    @staticmethod
    def _extract_reasoning(text: str) -> str:
        """Pull out the chain-of-thought between ``<think>`` and ``</think>``."""
        if "<think>" in text and "</think>" in text:
            start = text.index("<think>") + len("<think>")
            end = text.index("</think>")
            return text[start:end].strip()
        return text.strip()

    def _parse_command(self, text: str) -> int:
        """Parse model output text into a command index."""
        text_lower = text.lower().strip()
        if "</think>" in text_lower:
            text_lower = text_lower.split("</think>")[-1].strip()
        for i, cmd in enumerate(self.commands):
            if cmd.lower() in text_lower:
                return i
        return 0

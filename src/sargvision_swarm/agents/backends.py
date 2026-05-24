"""Pluggable LLM backends. Default to MockBackend so the demo runs without API keys.

Set SARGVISION_LLM_BACKEND=anthropic|ollama|mock to switch.
"""

from __future__ import annotations

import json
import os
import random
from typing import Protocol


class LLMBackend(Protocol):
    """Minimal interface: take a system + user message, return text."""

    def complete(self, system: str, user: str, max_tokens: int = 256) -> str: ...


class MockBackend:
    """Deterministic mock — returns a JSON-shaped decision derived from prompt hash.

    Use for demos without API keys + for tests.
    """

    def __init__(self, seed: int | None = 0) -> None:
        self.rng = random.Random(seed)

    def complete(self, system: str, user: str, max_tokens: int = 256) -> str:
        # Heuristic: parse drone id + neighbor count out of the user prompt and
        # emit a plausible JSON decision the agent layer can consume.
        intents = [
            "hold_formation",
            "advance_to_goal",
            "yield_to_neighbor",
            "rotate_role",
            "report_health",
        ]
        intent = self.rng.choice(intents)
        rationale = "mock-backend canned response (no LLM available)"
        return json.dumps({"intent": intent, "rationale": rationale})


class AnthropicBackend:
    """Anthropic Claude API. Requires ANTHROPIC_API_KEY in env."""

    def __init__(self, model: str = "claude-haiku-4-5") -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise RuntimeError("anthropic SDK not installed. `pip install '.[llm]'`") from e
        from anthropic import Anthropic

        self.client = Anthropic()
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 256) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in msg.content if block.type == "text")


class OllamaBackend:
    """Local Ollama runtime. Requires `ollama` running on localhost:11434."""

    def __init__(self, model: str = "qwen2.5:3b-instruct-q4_K_M") -> None:
        import urllib.request

        self._http = urllib.request
        self.model = model
        self.url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

    def complete(self, system: str, user: str, max_tokens: int = 256) -> str:
        import urllib.error
        import urllib.request

        body = json.dumps(
            {
                "model": self.model,
                "system": system,
                "prompt": user,
                "stream": False,
                "options": {"num_predict": max_tokens},
            }
        ).encode()
        req = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError) as e:
            raise RuntimeError(f"Ollama unreachable at {self.url}: {e}") from e
        return payload.get("response", "")


def make_backend(name: str | None = None) -> LLMBackend:
    """Factory. Picks MockBackend by default — works without external services."""
    backend = (name or os.getenv("SARGVISION_LLM_BACKEND") or "mock").lower()
    if backend == "anthropic":
        return AnthropicBackend()
    if backend == "ollama":
        return OllamaBackend()
    if backend == "mock":
        return MockBackend()
    raise ValueError(f"unknown backend: {backend}")

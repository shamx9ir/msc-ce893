"""Minimal OpenAI-compatible chat client (works with Ollama, vLLM, llama.cpp server, LM Studio)."""
import time
from dataclasses import dataclass, field

import requests


@dataclass
class LLMResponse:
    message: dict
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


@dataclass
class LLMClient:
    model: str
    base_url: str = "http://localhost:11434/v1"   # Ollama's OpenAI-compatible endpoint
    api_key: str = "not-needed"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: float = 300
    extra: dict = field(default_factory=dict)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        payload = {"model": self.model, "messages": messages,
                   "temperature": self.temperature, "max_tokens": self.max_tokens, **self.extra}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        t0 = time.time()
        r = requests.post(f"{self.base_url.rstrip('/')}/chat/completions", json=payload,
                          headers={"Authorization": f"Bearer {self.api_key}"}, timeout=self.timeout)
        r.raise_for_status()
        body = r.json()
        usage = body.get("usage") or {}
        return LLMResponse(message=body["choices"][0]["message"],
                           prompt_tokens=usage.get("prompt_tokens", 0),
                           completion_tokens=usage.get("completion_tokens", 0),
                           latency_s=time.time() - t0)

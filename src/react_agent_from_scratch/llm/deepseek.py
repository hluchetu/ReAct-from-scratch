from __future__ import annotations

import os

from openai import OpenAI

from .llm_client import LLMClient
from .model import ModelSettings


class DeepSeekClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        thinking: bool = True,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.thinking = thinking

    def generate(self, prompt: str) -> str:
        extra_body = (
            {"thinking": {"type": "enabled"}}
            if self.thinking
            else {"thinking": {"type": "disabled"}}
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            extra_body=extra_body,
            temperature=0,
            system=False,
            reasoning_effort="high" if self.thinking else None,
            stream=False,
        )

        return response.choices[0].message.content or ""


class DeepSeekProvider:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        thinking: bool = True,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._thinking = thinking

    def get_model(self, model_name: str, settings: ModelSettings) -> DeepSeekClient:
        return DeepSeekClient(
            model=model_name,
            api_key=self._api_key,
            base_url=self._base_url,
            thinking=self._thinking,
        )

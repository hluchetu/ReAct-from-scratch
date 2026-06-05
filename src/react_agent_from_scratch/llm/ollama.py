from __future__ import annotations

import json
import urllib.request

from pydantic import BaseModel

from .model import Model, ModelSettings


class OllamaModel:
    def __init__(self, name: str, base_url: str, settings: ModelSettings) -> None:
        self.name = name
        self.settings = settings
        self._base_url = base_url.rstrip("/")

    def generate(
        self,
        prompt: str,
        response_format: type[BaseModel] | None = None,
    ) -> str:
        payload: dict = {
            "model": self.name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.settings.temperature,
        }
        if response_format is not None:
            payload["format"] = response_format.model_json_schema()

        body = json.dumps(payload).encode()

        request = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
            response_data = json.loads(response.read().decode())

        return response_data.get("response", "").strip()


class OllamaProvider:
    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self._base_url = base_url

    def get_model(self, model_name: str, settings: ModelSettings) -> Model:
        return OllamaModel(name=model_name, base_url=self._base_url, settings=settings)

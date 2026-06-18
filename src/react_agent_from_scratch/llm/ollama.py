from __future__ import annotations

import json
import urllib.request

from pydantic import BaseModel

from ..message import AIMessage
from ..message import HumanMessage
from ..message import Message
from ..message import SystemMessage
from ..message import ToolMessage
from .model import ChatModel
from .model import ModelSettings


class OllamaModel:
    def __init__(self, name: str, base_url: str, settings: ModelSettings) -> None:
        self.name = name
        self.settings = settings
        self._base_url = base_url.rstrip("/")

    def invoke(
        self,
        messages: list[Message],
        response_format: type[BaseModel] | None = None,
    ) -> AIMessage:
        payload: dict = {
            "model": self.name,
            "messages": [self._to_provider_message(message) for message in messages],
            "options": {
                "temperature": self.settings.temperature,
                "num_predict": self.settings.max_tokens,
            },
            "stream": False,
        }
        if response_format is not None:
            payload["format"] = response_format.model_json_schema()

        body = json.dumps(payload).encode()

        request = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
            response_data = json.loads(response.read().decode())

        return AIMessage(content=response_data.get("message", {}).get("content", "").strip())

    def _to_provider_message(self, message: Message) -> dict[str, str]:
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, ToolMessage):
            role = "tool"
        else:
            raise TypeError(f"Unsupported message type: {type(message).__name__}")

        return {
            "role": role,
            "content": message.content,
        }


class OllamaProvider:
    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self._base_url = base_url

    def get_model(self, model_name: str, settings: ModelSettings) -> ChatModel:
        return OllamaModel(name=model_name, base_url=self._base_url, settings=settings)

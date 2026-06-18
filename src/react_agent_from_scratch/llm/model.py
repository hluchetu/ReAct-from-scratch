from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from ..message import AIMessage
from ..message import Message


class ModelSettings(BaseModel):
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: float = 120.0
    extra: dict[str, object] = Field(default_factory=dict)


class ChatModel(Protocol):
    name: str
    settings: ModelSettings

    def invoke(
        self,
        messages: list[Message],
        response_format: type[BaseModel] | None = None,
    ) -> AIMessage: ...


class ChatModelProvider(Protocol):
    def get_model(self, model_name: str, settings: ModelSettings) -> ChatModel: ...

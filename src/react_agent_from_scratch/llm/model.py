from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field


class ModelSettings(BaseModel):
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: float = 120.0
    extra: dict[str, object] = Field(default_factory=dict)


class Model(Protocol):
    name: str
    settings: ModelSettings

    def generate(
        self,
        prompt: str,
        response_format: type[BaseModel] | None = None,
    ) -> str: ...


class ModelProvider(Protocol):
    def get_model(self, model_name: str, settings: ModelSettings) -> Model: ...

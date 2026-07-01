from __future__ import annotations

from sophons.integrations.models import (
    AnthropicProvider,
    DeepSeekProvider,
    ModelSettings,
    OllamaProvider,
)
from sophons.models import ChatModel
from sophons.tools import Tool

from react_agent_from_scratch.config import settings


class ModelProviderRegistry:
    def __init__(self) -> None:
        self.providers = {
            "ollama": OllamaProvider(base_url=settings.ollama_base_url),
            "deepseek": DeepSeekProvider(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url),
            "anthropic": AnthropicProvider(api_key=settings.anthropic_api_key),
        }

    def get_model(
        self,
        provider_name: str,
        model_name: str,
        settings: ModelSettings,
        tools: list[Tool] | None = None,
    ) -> ChatModel:
        provider = self.providers.get(provider_name)
        if not provider:
            raise ValueError(
                f"Unknown provider '{provider_name}'. "
                f"Available: {', '.join(self.providers)}"
            )
        return provider.get_model(model_name, settings, tools=tools)

    @staticmethod
    def _parse_model_ref(model_ref: str) -> tuple[str, str]:
        if ":" not in model_ref:
            raise ValueError(
                f"Model reference must be 'provider:model', e.g. 'deepseek:deepseek-chat'. Got: {model_ref!r}"
            )
        provider, model = model_ref.split(":", maxsplit=1)
        if not provider or not model:
            raise ValueError("Both provider and model name must be non-empty.")
        return provider, model

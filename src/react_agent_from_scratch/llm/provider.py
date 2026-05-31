from __future__ import annotations

from ..config import settings
from .deepseek import DeepSeekProvider
from .model import Model, ModelProvider, ModelSettings
from .ollama import OllamaProvider


class ModelProviderRegistry:
    def __init__(self) -> None:
        self.providers: dict[str, ModelProvider] = {
            "ollama": OllamaProvider(base_url=settings.ollama_base_url),
            "deepseek": DeepSeekProvider(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
            ),
        }

    def register_provider(self, name: str, provider: ModelProvider) -> None:
        self.providers[name] = provider

    def get_model(
        self, provider_name: str, model_name: str, settings: ModelSettings
    ) -> Model:
        provider = self.providers.get(provider_name)
        if not provider:
            raise ValueError(f"Model provider '{provider_name}' not found.")
        return provider.get_model(model_name, settings)

    @staticmethod
    def _parse_model_ref(model_ref: str) -> tuple[str, str]:
        if ":" not in model_ref:
            raise ValueError(
                "Model reference must use the format 'provider:model', "
                "for example 'ollama:qwen3:4b' or 'deepseek:deepseek-v4-flash'."
            )

        provider_name, model_name = model_ref.split(":", maxsplit=1)

        if not provider_name or not model_name:
            raise ValueError(
                "Model reference must include both provider and model name."
            )

        return provider_name, model_name

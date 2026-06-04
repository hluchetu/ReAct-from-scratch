from __future__ import annotations
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ObservabilityProvider = Literal["none", "langfuse", "phoenix"]


class ObservabilityConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = Field(
        default=False,
        validation_alias="OBSERVABILITY_ENABLED",
    )

    provider: ObservabilityProvider = Field(
        default="none",
        validation_alias="OBSERVABILITY_PROVIDER",
    )
    langfuse_public_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LANGFUSE_PUBLIC_KEY", "LANGFUSE_API_KEY"),
    )
    langfuse_secret_key: str | None = Field(
        default=None,
        validation_alias="LANGFUSE_SECRET_KEY",
    )

    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        validation_alias="LANGFUSE_HOST",
    )

    phoenix_endpoint: str | None = Field(
        default="http://localhost:6006/v1/traces",
        validation_alias="PHOENIX_ENDPOINT",
    )

    @property
    def is_langfuse_ready(self) -> bool:
        return (
            self.enabled
            and self.provider == "langfuse"
            and bool(self.langfuse_public_key)
            and bool(self.langfuse_secret_key)
        )

    @property
    def is_phoenix_ready(self) -> bool:
        return (
            self.enabled and self.provider == "phoenix" and bool(self.phoenix_endpoint)
        )

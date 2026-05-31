from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    deepseek_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    deepseek_base_url: str = "https://api.deepseek.com/v1"


settings = Settings()

from __future__ import annotations

from openai import OpenAI
from pydantic import BaseModel

from ..message import AIMessage
from ..message import HumanMessage
from ..message import Message
from ..message import SystemMessage
from ..message import ToolMessage
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

    def invoke(
        self,
        messages: list[Message],
        response_format: type[BaseModel] | None = None,
    ) -> AIMessage:
        extra_body = (
            {"thinking": {"type": "enabled"}}
            if self.thinking
            else {"thinking": {"type": "disabled"}}
        )
        kwargs = dict(
            model=self.model,
            messages=[self._to_provider_message(message) for message in messages],
            extra_body=extra_body,
            temperature=0,
            reasoning_effort="high" if self.thinking else None,
            stream=False,
        )
        if response_format is not None and not self.thinking:
            kwargs["response_format"] = response_format

        response = self.client.beta.chat.completions.parse(**kwargs)
        return AIMessage(content=response.choices[0].message.content or "")

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

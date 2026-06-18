from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .message import AIMessage
from .message import HumanMessage
from .message import Message
from .message import SystemMessage
from .message import ToolCall
from .message import ToolMessage


class ChatContext(BaseModel):
    messages: list[Message] = Field(default_factory=list)

    def add_system_message(self, content: str) -> None:
        self.messages.append(SystemMessage(content=content))

    def add_human_message(self, content: str) -> None:
        self.messages.append(HumanMessage(content=content))

    def add_ai_message(self, content: str, tool_calls: list[ToolCall] = None) -> None:
        if tool_calls is None:
            tool_calls = []
        self.messages.append(AIMessage(content=content, tool_calls=tool_calls))

    def add_tool_message(
        self,
        tool_call_id: str,
        content: str,
        status: Literal["success", "error"] = "success",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.messages.append(
            ToolMessage(
                tool_call_id=tool_call_id,
                content=content,
                status=status,
                metadata=metadata or {},
            )
        )

    def format_for_model(self) -> str:
        formatted_messages: list[str] = []
        for message in self.messages:
            if isinstance(message, SystemMessage):
                formatted_messages.append(f"[SYSTEM] {message.content}")
            elif isinstance(message, HumanMessage):
                formatted_messages.append(f"[HUMAN] {message.content}")
            elif isinstance(message, AIMessage):
                formatted_messages.append(f"[AI] {message.content}")
                for tool_call in message.tool_calls:
                    formatted_messages.append(
                        f"[TOOL_CALL] {tool_call.name}({tool_call.args})"
                    )
            elif isinstance(message, ToolMessage):
                formatted_messages.append(
                    f"[TOOL_RESULT:{message.status}] "
                    f"{message.tool_call_id}: {message.content}"
                )
        return "\n\n".join(formatted_messages)

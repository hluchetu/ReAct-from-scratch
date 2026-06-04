from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, Field


class SystemMessage(BaseModel):
    type: Literal["system"] = "system"
    content: str


class UserMessage(BaseModel):
    type: Literal["user"] = "user"
    content: str


class ToolCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any]


class AssistantMessage(BaseModel):
    type: Literal["assistant"] = "assistant"
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolMessage(BaseModel):
    type: Literal["tool"] = "tool"
    tool_call_id: str
    content: str


Message = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]


class ChatContext(BaseModel):
    messages: list[Message] = Field(default_factory=list)

    def add_system_message(self, content: str) -> None:
        self.messages.append(SystemMessage(content=content))

    def add_user_message(self, content: str) -> None:
        self.messages.append(UserMessage(content=content))

    def add_assistant_message(
        self, content: str, tool_calls: list[ToolCall] = None
    ) -> None:
        if tool_calls is None:
            tool_calls = []
        self.messages.append(AssistantMessage(content=content, tool_calls=tool_calls))

    def add_tool_message(self, tool_call_id: str, content: str) -> None:
        self.messages.append(ToolMessage(tool_call_id=tool_call_id, content=content))

    def format_for_model(self) -> str:
        formatted_messages: list[str] = []
        for message in self.messages:
            if isinstance(message, SystemMessage):
                formatted_messages.append(f"[SYSTEM] {message.content}")
            elif isinstance(message, UserMessage):
                formatted_messages.append(f"[USER] {message.content}")
            elif isinstance(message, AssistantMessage):
                formatted_messages.append(f"[ASSISTANT] {message.content}")
                for tool_call in message.tool_calls:
                    formatted_messages.append(
                        f"[TOOL_CALL] {tool_call.name}({tool_call.args})"
                    )
            elif isinstance(message, ToolMessage):
                formatted_messages.append(
                    f"[TOOL_RESULT] {message.tool_call_id}: {message.content}"
                )
        return "\n\n".join(formatted_messages)

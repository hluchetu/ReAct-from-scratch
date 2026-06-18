from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, Field


class SystemMessage(BaseModel):
    type: Literal["system"] = "system"
    content: str


class HumanMessage(BaseModel):
    type: Literal["human"] = "human"
    content: str


class ToolCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any]


class AIMessage(BaseModel):
    type: Literal["ai"] = "ai"
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolMessage(BaseModel):
    type: Literal["tool"] = "tool"
    tool_call_id: str
    content: str
    status: Literal["success", "error"] = "success"
    metadata: dict[str, Any] = Field(default_factory=dict)


Message = Union[SystemMessage, HumanMessage, AIMessage, ToolMessage]

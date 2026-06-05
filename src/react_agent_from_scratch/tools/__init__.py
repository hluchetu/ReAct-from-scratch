from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..context import ToolCall, ToolMessage


class ToolInputSchema(BaseModel):
    type: str = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class ToolExecutionResult(BaseModel):
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: Literal["success", "error"] = "success"


class Tool(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    input_schema: ToolInputSchema
    run: Callable[[dict[str, Any]], ToolExecutionResult]


class ToolRegistry(BaseModel):
    tools: dict[str, Tool] = Field(default_factory=dict)

    def register_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def get(self, tool_name: str) -> Tool | None:
        return self.tools.get(tool_name)

    def format_for_model(self) -> str:
        if not self.tools:
            return "No tools registered."

        tool_descriptions: list[str] = []

        for tool in self.tools.values():
            tool_descriptions.append(
                f"- {tool.name}: {tool.description}\n"
                f"  input_schema: {tool.input_schema.model_dump()}"
            )

        return "\n\n".join(tool_descriptions)

    def execute_tool(self, tool_call: ToolCall) -> ToolMessage:
        tool = self.get(tool_call.name)

        if tool is None:
            return ToolMessage(
                tool_call_id=tool_call.id,
                content=f"Error: Tool '{tool_call.name}' not found.",
                status="error",
                metadata={"recoverable": True, "reason": "unknown_tool"},
            )

        try:
            result = tool.run(tool_call.args)
            return ToolMessage(
                tool_call_id=tool_call.id,
                content=result.content,
                status=result.status,
                metadata=result.metadata,
            )
        except Exception as exc:
            return ToolMessage(
                tool_call_id=tool_call.id,
                content=f"Error executing tool '{tool_call.name}': {exc}",
                status="error",
                metadata={"recoverable": True, "reason": "tool_exception"},
            )

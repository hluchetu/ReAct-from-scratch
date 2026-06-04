from __future__ import annotations

from react_agent_from_scratch.context import ToolCall
from react_agent_from_scratch.tools import (
    Tool,
    ToolExecutionResult,
    ToolInputSchema,
    ToolRegistry,
)


def test_known_tool_executes_successfully() -> None:
    registry = ToolRegistry()
    registry.register_tool(
        Tool(
            name="echo",
            description="Echo the input text.",
            input_schema=ToolInputSchema(
                properties={"text": {"type": "string"}},
                required=["text"],
            ),
            run=lambda args: ToolExecutionResult(
                content=args["text"],
                status="success",
                metadata={"source": "echo_tool"},
            ),
        )
    )

    result = registry.execute_tool(
        ToolCall(id="call_1", name="echo", args={"text": "hello"})
    )

    assert result.tool_call_id == "call_1"
    assert result.content == "hello"
    assert result.status == "success"
    assert result.metadata == {"source": "echo_tool"}


def test_unknown_tool_returns_recoverable_error() -> None:
    registry = ToolRegistry()

    result = registry.execute_tool(
        ToolCall(id="call_1", name="missing_tool", args={})
    )

    assert result.tool_call_id == "call_1"
    assert result.status == "error"
    assert "not found" in result.content
    assert result.metadata["recoverable"] is True
    assert result.metadata["reason"] == "unknown_tool"


def test_tool_exception_returns_recoverable_error() -> None:
    registry = ToolRegistry()

    def failing_tool(args: dict) -> ToolExecutionResult:
        raise RuntimeError("network timeout")

    registry.register_tool(
        Tool(
            name="failing_tool",
            description="Always fails.",
            input_schema=ToolInputSchema(),
            run=failing_tool,
        )
    )

    result = registry.execute_tool(
        ToolCall(id="call_1", name="failing_tool", args={})
    )

    assert result.tool_call_id == "call_1"
    assert result.status == "error"
    assert "network timeout" in result.content
    assert result.metadata["recoverable"] is True
    assert result.metadata["reason"] == "tool_exception"


def test_tool_result_status_and_metadata_are_preserved() -> None:
    registry = ToolRegistry()
    registry.register_tool(
        Tool(
            name="soft_fail",
            description="Returns an error result without raising.",
            input_schema=ToolInputSchema(),
            run=lambda args: ToolExecutionResult(
                content="No results found.",
                status="error",
                metadata={"recoverable": True, "reason": "empty_result"},
            ),
        )
    )

    result = registry.execute_tool(ToolCall(id="call_1", name="soft_fail", args={}))

    assert result.status == "error"
    assert result.content == "No results found."
    assert result.metadata["recoverable"] is True
    assert result.metadata["reason"] == "empty_result"

from __future__ import annotations

from react_agent_from_scratch.llm.agent import ReActAgent
from react_agent_from_scratch.llm.model import ModelSettings
from react_agent_from_scratch.run import RunConfig, RunLimits
from react_agent_from_scratch.tools import (
    Tool,
    ToolExecutionResult,
    ToolInputSchema,
    ToolRegistry,
)


class AlwaysToolModel:
    name = "fake-always-tool"
    settings = ModelSettings()

    def generate(self, prompt: str) -> str:
        return (
            "Thought: I should use a tool.\n"
            'Tool Call: {"name": "echo", "args": {"text": "hello"}}'
        )


def build_test_agent(config: RunConfig) -> ReActAgent:
    tools = ToolRegistry()
    tools.register_tool(
        Tool(
            name="echo",
            description="Echo input text.",
            input_schema=ToolInputSchema(
                properties={"text": {"type": "string"}},
                required=["text"],
            ),
            run=lambda args: ToolExecutionResult(content=args["text"]),
        )
    )

    return ReActAgent(
        model=AlwaysToolModel(),
        tools=tools,
        system_prompt="You are a test agent.",
        config=config,
    )


def test_stops_at_max_steps() -> None:
    agent = build_test_agent(
        RunConfig(
            limits=RunLimits(
                max_steps=2,
                max_model_calls=10,
                max_tool_calls=10,
            )
        )
    )

    result = agent.run("Use the tool.")

    assert result.success is False
    assert result.stop_reason == "max_steps"
    assert result.steps == 2
    assert result.model_calls == 2
    assert result.tool_calls == 2


def test_stops_at_max_model_calls() -> None:
    agent = build_test_agent(
        RunConfig(
            limits=RunLimits(
                max_steps=10,
                max_model_calls=2,
                max_tool_calls=10,
            )
        )
    )

    result = agent.run("Use the tool.")

    assert result.success is False
    assert result.stop_reason == "max_model_calls"
    assert result.model_calls == 2


def test_stops_at_max_tool_calls() -> None:
    agent = build_test_agent(
        RunConfig(
            limits=RunLimits(
                max_steps=10,
                max_model_calls=10,
                max_tool_calls=2,
            )
        )
    )

    result = agent.run("Use the tool.")

    assert result.success is False
    assert result.stop_reason == "max_tool_calls"
    assert result.tool_calls == 2

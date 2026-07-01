from __future__ import annotations

from sophons.agents import Agent
from sophons.models import Message
from sophons.tools import tool

from react_agent_from_scratch.llm.ollama import _normalize_tool_calls


class ToolCallingModel:
    def __init__(self) -> None:
        self.calls = 0
        self.messages_seen: list[list[Message]] = []

    def invoke(self, messages: list[Message]) -> Message:
        self.calls += 1
        self.messages_seen.append(messages)

        if self.calls == 1:
            return Message(
                role="assistant",
                content="",
                metadata={
                    "tool_calls": [
                        {
                            "tool_use_id": "call_1",
                            "name": "add",
                            "input": {"a": 2, "b": 3},
                        }
                    ]
                },
            )

        return Message(role="assistant", content="The answer is 5.")


def test_sophons_agent_executes_native_tool_call() -> None:
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    model = ToolCallingModel()
    agent = Agent(
        model=model,
        tools=[add],
        system_prompt="You are a helpful agent.",
    )

    result = agent.run_sync("What is 2 + 3?")

    assert result.success is True
    assert result.message == "The answer is 5."
    assert result.metrics.model_calls == 2
    assert result.metrics.tool_calls == 1
    assert result.tool_results[0].content == '{"result": 5}'

    second_call_messages = model.messages_seen[1]
    assert second_call_messages[-1].role == "tool"
    assert second_call_messages[-1].content == '{"result": 5}'


def test_ollama_tool_calls_are_normalized_for_sophons() -> None:
    raw_tool_calls = [
        {
            "id": "call_1",
            "function": {
                "name": "web_search",
                "arguments": '{"query": "latest Python release"}',
            },
        }
    ]

    assert _normalize_tool_calls(raw_tool_calls) == [
        {
            "tool_use_id": "call_1",
            "name": "web_search",
            "input": {"query": "latest Python release"},
        }
    ]

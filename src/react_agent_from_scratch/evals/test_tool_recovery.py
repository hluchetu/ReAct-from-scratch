from __future__ import annotations

from react_agent_from_scratch.llm.agent import ReActAgent
from react_agent_from_scratch.llm.model import ModelSettings
from react_agent_from_scratch.run import RunConfig, RunLimits
from react_agent_from_scratch.tools import ToolRegistry


class UnknownToolThenFinalModel:
    name = "fake-unknown-tool-then-final"
    settings = ModelSettings()

    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)

        if self.calls == 1:
            return (
                "Thought: I should use a tool.\n"
                'Tool Call: {"name": "missing_tool", "args": {"query": "MCP"}}'
            )

        return "Final Answer: I corrected after seeing the missing tool observation."


def test_unknown_tool_error_becomes_observation_and_agent_recovers() -> None:
    model = UnknownToolThenFinalModel()

    agent = ReActAgent(
        model=model,
        tools=ToolRegistry(),
        system_prompt="You are a test agent.",
        config=RunConfig(
            limits=RunLimits(
                max_steps=3,
                max_model_calls=3,
                max_tool_calls=3,
            )
        ),
    )

    result = agent.run("Use a tool if needed.")

    assert result.success is True
    assert result.stop_reason == "final_answer"
    assert result.content == "I corrected after seeing the missing tool observation."
    assert result.model_calls == 2
    assert result.tool_calls == 1

    second_prompt = model.prompts[1]

    assert "[TOOL_RESULT:error]" in second_prompt
    assert "Tool 'missing_tool' not found" in second_prompt

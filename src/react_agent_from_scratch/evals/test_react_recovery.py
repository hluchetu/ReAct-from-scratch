from __future__ import annotations

from react_agent_from_scratch.llm.agent import ReActAgent
from react_agent_from_scratch.llm.model import ModelSettings
from react_agent_from_scratch.run import RunConfig, RunLimits
from react_agent_from_scratch.tools import ToolRegistry


class InvalidThenFinalModel:
    name = "fake-invalid-then-final"
    settings = ModelSettings()

    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)

        if self.calls == 1:
            return "I should probably search first, but I forgot the format."

        return "Final Answer: I corrected my format after reading the observation."


def test_parser_failure_becomes_observation_and_agent_recovers() -> None:
    model = InvalidThenFinalModel()

    agent = ReActAgent(
        model=model,
        tools=ToolRegistry(),
        system_prompt="You are a test agent.",
        config=RunConfig(
                limits=RunLimits(
                    max_steps=3,
                    max_model_calls=3,
                    max_tool_calls=1,
                )
            ),
        )

    result = agent.run("Answer using the correct ReAct format.")

    assert result.success is True
    assert result.stop_reason == "final_answer"
    assert result.content == "I corrected my format after reading the observation."
    assert result.model_calls == 2

    second_prompt = model.prompts[1]

    assert "[TOOL_RESULT:success] parser:" in second_prompt
    assert "Model output did not match the expected ReAct format" in second_prompt

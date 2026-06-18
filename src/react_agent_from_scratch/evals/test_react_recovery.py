from __future__ import annotations

from react_agent_from_scratch.llm.agent import ReActAgent
from react_agent_from_scratch.llm.model import ModelSettings
from react_agent_from_scratch.message import AIMessage
from react_agent_from_scratch.message import Message
from react_agent_from_scratch.run import RunConfig, RunLimits
from react_agent_from_scratch.tools import ToolRegistry


class InvalidThenFinalModel:
    name = "fake-invalid-then-final"
    settings = ModelSettings()

    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    def invoke(self, messages: list[Message], response_format=None) -> AIMessage:
        import json
        self.calls += 1
        prompt = messages[-1].content
        self.prompts.append(prompt)

        if self.calls == 1:
            return AIMessage(
                content="I should probably search first, but I forgot the format."
            )

        return AIMessage(
            content=json.dumps({
                "thought": "I corrected my format after reading the observation.",
                "type": "final_answer",
                "answer": "I corrected my format after reading the observation.",
            })
        )


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
    assert "not valid JSON" in second_prompt

from __future__ import annotations

from ..context import ChatContext
from ..output_parser import FinalAnswer, ParsedToolCall, ReActOutputParser
from ..run import AgentRunResult, RunConfig, RunState
from ..tools import ToolRegistry
from .model import Model


class ReActAgent:
    def __init__(
        self,
        model: Model,
        tools: ToolRegistry,
        system_prompt: str,
        config: RunConfig = RunConfig(),
    ) -> None:
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.config = config
        self.output_parser = ReActOutputParser()

    def run(self, user_message: str) -> AgentRunResult:
        context = ChatContext()
        context.add_system_message(self.system_prompt)
        context.add_user_message(user_message)

        state = RunState()
        limits = self.config.limits

        for _ in range(limits.max_steps):
            if state.elapsed_seconds() > limits.max_runtime_seconds:
                state.mark_stopped("max_runtime_seconds")
                break

            if state.model_call_count >= limits.max_model_calls:
                state.mark_stopped("max_model_calls")
                break

            if state.tool_call_count >= limits.max_tool_calls:
                state.mark_stopped("max_tool_calls")
                break

            state.step_count += 1

            prompt = context.format_for_model()
            model_output = self.model.generate(prompt)
            state.model_call_count += 1

            parsed_output = self.output_parser.parse(model_output)

            if isinstance(parsed_output, FinalAnswer):
                state.mark_stopped("final_answer")
                return AgentRunResult(
                    content=parsed_output.content,
                    stop_reason=state.stop_reason,
                    success=True,
                    steps=state.step_count,
                    model_calls=state.model_call_count,
                    tool_calls=state.tool_call_count,
                )

            if isinstance(parsed_output, ParsedToolCall):
                context.add_assistant_message(
                    content=model_output,
                    tool_calls=[parsed_output.tool_call],
                )
                tool_message = self.tools.execute_tool(parsed_output.tool_call)
                state.tool_call_count += 1
                context.messages.append(tool_message)

        if state.stop_reason is None:
            state.mark_stopped("max_steps")

        return AgentRunResult(
            content="Agent stopped before reaching a final answer.",
            stop_reason=state.stop_reason,
            success=False,
            steps=state.step_count,
            model_calls=state.model_call_count,
            tool_calls=state.tool_call_count,
            error=state.error,
        )

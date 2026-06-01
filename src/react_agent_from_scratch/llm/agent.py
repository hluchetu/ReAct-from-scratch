from __future__ import annotations

from ..context import ChatContext
from ..output_parser import FinalAnswer, ParsedToolCall, ReActOutputParser
from ..tools import ToolRegistry
from .model import Model


class ReActAgent:
    def __init__(
        self,
        model: Model,
        tools: ToolRegistry,
        system_prompt: str,
        max_steps: int = 5,
    ) -> None:
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.output_parser = ReActOutputParser()

    def run(self, user_message: str) -> str:
        context = ChatContext()
        context.add_system_message(self.system_prompt)
        context.add_user_message(user_message)

        for _ in range(self.max_steps):
            prompt = context.format_for_model()
            model_output = self.model.generate(prompt)
            parsed_output = self.output_parser.parse(model_output)

            if isinstance(parsed_output, FinalAnswer):
                return parsed_output.content

            if isinstance(parsed_output, ParsedToolCall):
                context.add_assistant_message(
                    content=model_output,
                    tool_calls=[parsed_output.tool_call],
                )
                tool_message = self.tools.execute_tool(parsed_output.tool_call)
                context.messages.append(tool_message)

        return "I stopped because the agent reached the maximum number of steps."

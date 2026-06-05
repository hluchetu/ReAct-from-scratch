from __future__ import annotations

from ..context import ChatContext
from ..observability.tracing import (
    AgentTracer,
    NoOpTracer,
    SpanContext,
    SpanResult,
    TraceContext,
    TraceEvent,
)
from ..output_parser import (
    FinalAnswer,
    ParsedToolCall,
    ReActOutputParser,
    ParserFailure,
    ResponseFormat,
)
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
        tracer: AgentTracer | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.config = config
        self.tracer = tracer or NoOpTracer()
        self.output_parser = ReActOutputParser()

    def run(self, user_message: str) -> AgentRunResult:
        context = ChatContext()
        context.add_system_message(self.system_prompt)
        context.add_human_message(user_message)

        state = RunState()
        limits = self.config.limits

        self.tracer.start_trace(
            TraceContext(
                name="react-agent-run",
                input=user_message,
                tags=["react", "from-scratch"],
                metadata={
                    "model": self.model.name,
                    "max_steps": limits.max_steps,
                    "max_model_calls": limits.max_model_calls,
                    "max_tool_calls": limits.max_tool_calls,
                },
            )
        )

        try:
            result = self._run_loop(context, state)
            self.tracer.end_trace(
                SpanResult(
                    output=result.content,
                    error=result.error,
                    metadata={
                        "stop_reason": result.stop_reason,
                        "success": result.success,
                        "steps": result.steps,
                        "model_calls": result.model_calls,
                        "tool_calls": result.tool_calls,
                    },
                )
            )
            return result

        except Exception as exc:
            state.mark_error(str(exc))

            self.tracer.record_event(
                TraceEvent(
                    name="agent_error",
                    level="error",
                    message=str(exc),
                    metadata={"step": state.step_count},
                )
            )

            result = AgentRunResult(
                content="Agent failed while running.",
                stop_reason=state.stop_reason,
                success=False,
                steps=state.step_count,
                model_calls=state.model_call_count,
                tool_calls=state.tool_call_count,
                error=str(exc),
            )

            self.tracer.end_trace(
                SpanResult(
                    output=result.content,
                    error=result.error,
                    metadata={"stop_reason": result.stop_reason},
                )
            )

            return result

    def _run_loop(self, context: ChatContext, state: RunState) -> AgentRunResult:
        limits = self.config.limits

        for _ in range(limits.max_steps):
            if state.elapsed_seconds() > limits.max_runtime_seconds:
                state.mark_stopped("max_runtime_seconds")
                self._record_stop_event(state)
                break

            if state.model_call_count >= limits.max_model_calls:
                state.mark_stopped("max_model_calls")
                self._record_stop_event(state)
                break

            if state.tool_call_count >= limits.max_tool_calls:
                state.mark_stopped("max_tool_calls")
                self._record_stop_event(state)
                break

            state.step_count += 1

            prompt = context.format_for_model()

            self.tracer.start_span(
                SpanContext(
                    name="model.generate",
                    span_type="model",
                    input=prompt,
                    metadata={
                        "step": state.step_count,
                        "model": self.model.name,
                    },
                )
            )

            model_output = self.model.generate(prompt, response_format=ResponseFormat)
            state.model_call_count += 1

            self.tracer.end_span(
                SpanResult(
                    output=model_output,
                    metadata={
                        "step": state.step_count,
                        "model_call_count": state.model_call_count,
                    },
                )
            )

            self.tracer.start_span(
                SpanContext(
                    name="parse.model_output",
                    span_type="parser",
                    input=model_output,
                    metadata={"step": state.step_count},
                )
            )

            parsed_output = self.output_parser.parse(model_output)

            self.tracer.end_span(
                SpanResult(
                    output=parsed_output.model_dump(),
                    metadata={"parsed_type": parsed_output.type},
                )
            )

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
            if isinstance(parsed_output, ParserFailure):
                self.tracer.record_event(
                    TraceEvent(
                        name="parser_failure",
                        level="error",
                        message=parsed_output.message,
                        metadata={
                            "step": state.step_count,
                            "raw_output": parsed_output.raw_output,
                        },
                    )
                )
                context.add_ai_message(content=parsed_output.raw_output)
                context.add_tool_message(
                    tool_call_id="parser",
                    content=parsed_output.message,
                )

                continue

            if isinstance(parsed_output, ParsedToolCall):
                context.add_ai_message(
                    content=model_output,
                    tool_calls=[parsed_output.tool_call],
                )

                self.tracer.start_span(
                    SpanContext(
                        name="tool.execute",
                        span_type="tool",
                        input=parsed_output.tool_call.args,
                        metadata={
                            "step": state.step_count,
                            "tool_name": parsed_output.tool_call.name,
                            "tool_call_id": parsed_output.tool_call.id,
                        },
                    )
                )

                tool_message = self.tools.execute_tool(parsed_output.tool_call)
                state.tool_call_count += 1

                self.tracer.end_span(
                    SpanResult(
                        output=tool_message.content,
                        metadata={
                            "tool_call_count": state.tool_call_count,
                            "tool_call_id": tool_message.tool_call_id,
                        },
                    )
                )

                context.messages.append(tool_message)

        if state.stop_reason is None:
            state.mark_stopped("max_steps")
            self._record_stop_event(state)

        return AgentRunResult(
            content="Agent stopped before reaching a final answer.",
            stop_reason=state.stop_reason,
            success=False,
            steps=state.step_count,
            model_calls=state.model_call_count,
            tool_calls=state.tool_call_count,
            error=state.error,
        )

    def _record_stop_event(self, state: RunState) -> None:
        self.tracer.record_event(
            TraceEvent(
                name="run_stopped",
                level="warning",
                message=f"Run stopped because of {state.stop_reason}.",
                metadata={
                    "stop_reason": state.stop_reason,
                    "steps": state.step_count,
                    "model_calls": state.model_call_count,
                    "tool_calls": state.tool_call_count,
                },
            )
        )

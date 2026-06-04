from __future__ import annotations

from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from phoenix.otel import register

from .config import ObservabilityConfig
from .tracing import AgentTracer, SpanContext, SpanResult, TraceContext, TraceEvent


OPENINFERENCE_SPAN_TYPES = {
    "agent": "AGENT",
    "model": "LLM",
    "tool": "TOOL",
    "retriever": "RETRIEVER",
    "guardrail": "CHAIN",
    "parser": "CHAIN",
    "custom": "CHAIN",
}


class PhoenixTracer:
    def __init__(self, config: ObservabilityConfig) -> None:
        if not config.is_phoenix_ready:
            raise ValueError("Phoenix is not configured.")

        register(endpoint=config.phoenix_endpoint)

        self.tracer = trace.get_tracer("react-from-scratch")
        self._span_stack: list[Any] = []
        self._span_context_stack: list[Any] = []

    def start_trace(self, context: TraceContext) -> None:
        span_context = self.tracer.start_as_current_span(context.name)
        span = span_context.__enter__()

        span.set_attribute("openinference.span.kind", "AGENT")

        if context.input is not None:
            span.set_attribute("input.value", str(context.input))

        if context.user_id:
            span.set_attribute("user.id", context.user_id)

        if context.session_id:
            span.set_attribute("session.id", context.session_id)

        if context.tags:
            span.set_attribute("tags", ",".join(context.tags))

        for key, value in context.metadata.items():
            span.set_attribute(f"metadata.{key}", str(value))

        self._span_context_stack.append(span_context)
        self._span_stack.append(span)

    def start_span(self, context: SpanContext) -> None:
        span_context = self.tracer.start_as_current_span(context.name)
        span = span_context.__enter__()

        span.set_attribute(
            "openinference.span.kind",
            OPENINFERENCE_SPAN_TYPES[context.span_type],
        )

        if context.input is not None:
            span.set_attribute("input.value", str(context.input))

        for key, value in context.metadata.items():
            span.set_attribute(f"metadata.{key}", str(value))

        self._span_context_stack.append(span_context)
        self._span_stack.append(span)

    def end_span(self, result: SpanResult | None = None) -> None:
        if not self._span_stack or not self._span_context_stack:
            return

        span = self._span_stack.pop()
        span_context = self._span_context_stack.pop()

        if result is not None:
            if result.output is not None:
                span.set_attribute("output.value", str(result.output))

            for key, value in result.metadata.items():
                span.set_attribute(f"metadata.{key}", str(value))

            if result.error:
                span.set_status(Status(StatusCode.ERROR, result.error))
                span.record_exception(Exception(result.error))
            else:
                span.set_status(Status(StatusCode.OK))

        span_context.__exit__(None, None, None)

    def record_event(self, event: TraceEvent) -> None:
        if not self._span_stack:
            return

        current_span = self._span_stack[-1]
        current_span.add_event(
            name=event.name,
            attributes={
                "message": event.message or "",
                "level": event.level,
                **{
                    f"metadata.{key}": str(value)
                    for key, value in event.metadata.items()
                },
            },
        )

    def end_trace(self, result: SpanResult | None = None) -> None:
        while self._span_stack:
            self.end_span(result if len(self._span_stack) == 1 else None)


def create_phoenix_tracer(config: ObservabilityConfig) -> AgentTracer:
    return PhoenixTracer(config)

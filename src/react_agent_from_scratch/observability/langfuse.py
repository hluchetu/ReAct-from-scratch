from __future__ import annotations

from typing import Any

from langfuse import Langfuse, propagate_attributes

from .config import ObservabilityConfig
from .tracing import (
    AgentTracer,
    SpanContext,
    SpanResult,
    TraceContext,
    TraceEvent,
)


LANGFUSE_LEVELS = {
    "debug": "DEBUG",
    "info": "DEFAULT",
    "warning": "WARNING",
    "error": "ERROR",
}

LANGFUSE_SPAN_TYPES = {
    "agent": "agent",
    "model": "generation",
    "tool": "tool",
    "retriever": "retriever",
    "guardrail": "guardrail",
    "parser": "span",
    "custom": "span",
}


class LangfuseTracer:
    def __init__(self, config: ObservabilityConfig) -> None:
        if not config.is_langfuse_ready:
            raise ValueError("Langfuse is not configured.")

        self.client = Langfuse(
            public_key=config.langfuse_public_key,
            secret_key=config.langfuse_secret_key,
            host=config.langfuse_host,
        )

        self._attribute_context: Any | None = None
        self._trace_context: Any | None = None
        self._trace_observation: Any | None = None
        self._span_stack: list[tuple[Any, Any]] = []

    def start_trace(self, context: TraceContext) -> None:
        self._attribute_context = propagate_attributes(
            user_id=context.user_id,
            session_id=context.session_id,
            tags=context.tags,
            metadata=context.metadata,
        )
        self._attribute_context.__enter__()

        kwargs: dict[str, Any] = {
            "name": context.name,
            "as_type": "agent",
            "input": context.input,
            "metadata": context.metadata,
        }

        if context.trace_id:
            kwargs["trace_context"] = {"trace_id": context.trace_id}

        self._trace_context = self.client.start_as_current_observation(**kwargs)
        self._trace_observation = self._trace_context.__enter__()

    def start_span(self, context: SpanContext) -> None:
        kwargs: dict[str, Any] = {
            "name": context.name,
            "input": context.input,
            "metadata": context.metadata,
        }

        kwargs["as_type"] = LANGFUSE_SPAN_TYPES[context.span_type]
        if context.span_type == "model":
            model = context.metadata.get("model")
            if model:
                kwargs["model"] = model

        if context.span_id or context.parent_span_id:
            trace_context: dict[str, str] = {}
            if context.span_id:
                trace_context["span_id"] = context.span_id
            if context.parent_span_id:
                trace_context["parent_span_id"] = context.parent_span_id
            kwargs["trace_context"] = trace_context

        span_context = self.client.start_as_current_observation(**kwargs)
        span_observation = span_context.__enter__()

        self._span_stack.append((span_context, span_observation))

    def end_span(self, result: SpanResult | None = None) -> None:
        if not self._span_stack:
            return

        span_context, span_observation = self._span_stack.pop()

        if result is not None:
            span_observation.update(
                output=result.output,
                metadata=result.metadata,
                status_message=result.error,
            )

        span_context.__exit__(None, None, None)

    def record_event(self, event: TraceEvent) -> None:
        self.client.create_event(
            name=event.name,
            output=event.message,
            level=LANGFUSE_LEVELS[event.level],
            metadata=event.metadata,
        )

    def end_trace(self, result: SpanResult | None = None) -> None:
        while self._span_stack:
            self.end_span()

        if self._trace_observation is not None and result is not None:
            self._trace_observation.update(
                output=result.output,
                metadata=result.metadata,
                status_message=result.error,
            )

        if self._trace_context is not None:
            self._trace_context.__exit__(None, None, None)

        if self._attribute_context is not None:
            self._attribute_context.__exit__(None, None, None)

        self.client.flush()


def create_langfuse_tracer(config: ObservabilityConfig) -> AgentTracer:
    return LangfuseTracer(config)

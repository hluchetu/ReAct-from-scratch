from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

SpanType = Literal[
    "agent",
    "model",
    "tool",
    "retriever",
    "guardrail",
    "parser",
    "custom",
]


class TraceContext(BaseModel):
    trace_id: str | None = None
    name: str
    input: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SpanContext(BaseModel):
    span_id: str | None = None
    parent_span_id: str | None = None
    name: str
    span_type: SpanType = "custom"
    input: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    name: str
    message: str | None = None
    level: Literal["debug", "info", "warning", "error"] = "info"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SpanResult(BaseModel):
    output: Any | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentTracer(Protocol):
    def start_trace(self, context: TraceContext) -> None: ...

    def start_span(self, context: SpanContext) -> None: ...

    def end_span(self, result: SpanResult | None = None) -> None: ...

    def record_event(self, event: TraceEvent) -> None: ...

    def end_trace(self, result: SpanResult | None = None) -> None: ...


class NoOpTracer:
    def start_trace(self, context: TraceContext) -> None:
        pass

    def start_span(self, context: SpanContext) -> None:
        pass

    def end_span(self, result: SpanResult | None = None) -> None:
        pass

    def record_event(self, event: TraceEvent) -> None:
        pass

    def end_trace(self, result: SpanResult | None = None) -> None:
        pass

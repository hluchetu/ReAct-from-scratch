from __future__ import annotations

from react_agent_from_scratch.observability.phoenix import PhoenixTracer

from .config import ObservabilityConfig
from .langfuse import create_langfuse_tracer
from .tracing import AgentTracer, NoOpTracer


class TracerProvider:
    def __init__(self, config: ObservabilityConfig | None = None) -> None:
        self.config = config or ObservabilityConfig()

    def get_tracer(self) -> AgentTracer:
        if self.config.is_langfuse_ready:
            return create_langfuse_tracer(self.config)

        if self.config.is_phoenix_ready:
            return PhoenixTracer(self.config)

        return NoOpTracer()

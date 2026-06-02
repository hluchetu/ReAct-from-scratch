from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

StopReason = Literal[
    "final_answer",
    "max_steps",
    "max_model_calls",
    "max_tool_calls",
    "max_runtime_seconds",
    "error",
]


class RunLimits(BaseModel):
    max_steps: int = 10
    max_model_calls: int = 20
    max_tool_calls: int = 20
    max_runtime_seconds: float = 300.0


class RunState(BaseModel):
    started_at: float = Field(default_factory=time.monotonic)
    step_count: int = 0
    model_call_count: int = 0
    tool_call_count: int = 0
    stop_reason: StopReason | None = None
    error: str | None = None

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    def mark_stopped(self, reason: StopReason) -> None:
        self.stop_reason = reason

    def mark_error(self, error: str) -> None:
        self.error = error
        self.stop_reason = "error"


class RunConfig(BaseModel):
    limits: RunLimits = RunLimits()


class AgentRunResult(BaseModel):
    content: str
    stop_reason: StopReason
    success: bool
    steps: int
    model_calls: int
    tool_calls: int
    error: str | None = None

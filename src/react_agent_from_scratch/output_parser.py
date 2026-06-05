from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel

from .context import ToolCall


class ResponseFormat(BaseModel):
    thought: str
    type: Literal["tool_call", "final_answer"]
    tool_name: str | None = None
    args: dict[str, Any] | None = None
    answer: str | None = None


class FinalAnswer(BaseModel):
    type: Literal["final_answer"] = "final_answer"
    content: str


class ParsedToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    thought: str
    tool_call: ToolCall


class ParserFailure(BaseModel):
    type: Literal["parser_failure"] = "parser_failure"
    message: str
    raw_output: str


ParsedModelOutput = FinalAnswer | ParsedToolCall | ParserFailure


class ReActOutputParser:
    def parse(self, text: str) -> ParsedModelOutput:
        try:
            response = ResponseFormat.model_validate_json(text)
        except Exception:
            return ParserFailure(
                message=(
                    "Model output was not valid JSON matching the expected schema. "
                    "Expected a JSON object with 'thought', 'type', and either "
                    "'tool_name'/'args' or 'answer'."
                ),
                raw_output=text,
            )

        if response.type == "final_answer":
            return FinalAnswer(content=response.answer or response.thought)

        if response.type == "tool_call":
            if not response.tool_name:
                return ParserFailure(
                    message="Tool call is missing 'tool_name'.",
                    raw_output=text,
                )
            return ParsedToolCall(
                thought=response.thought,
                tool_call=ToolCall(
                    id=str(uuid4()),
                    name=response.tool_name,
                    args=response.args or {},
                ),
            )

        return ParserFailure(
            message=f"Unknown response type: {response.type}.",
            raw_output=text,
        )

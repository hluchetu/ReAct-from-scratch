from __future__ import annotations

import json
import re
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel

from .context import ToolCall


class FinalAnswer(BaseModel):
    type: Literal["final_answer"] = "final_answer"
    content: str


class ParsedToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    thought: str
    tool_call: ToolCall


ParsedModelOutput = FinalAnswer | ParsedToolCall


class ReActOutputParser:
    def parse(self, text: str) -> ParsedModelOutput:
        final_answer = self._parse_final_answer(text)
        if final_answer is not None:
            return final_answer
        tool_call = self._parse_tool_call(text)
        if tool_call is not None:
            return tool_call

        return FinalAnswer(content=text.strip())

    def _parse_final_answer(self, text: str) -> FinalAnswer | None:
        match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL)

        if match is None:
            return None

        return FinalAnswer(content=match.group(1).strip())

    def _parse_tool_call(self, text: str) -> ParsedToolCall | None:
        match = re.search(r"Thought:\s*(.*?)\s*Tool Call:\s*(\{.*\})", text, re.DOTALL)

        if match is None:
            return None

        thought = match.group(1).strip()
        tool_call_json = match.group(2).strip()

        try:
            tool_call_dict = json.loads(tool_call_json)
            tool_call = ToolCall(
                id=str(uuid4()),
                name=tool_call_dict["name"],
                args=tool_call_dict.get("args", {}),
            )
            return ParsedToolCall(thought=thought, tool_call=tool_call)
        except (json.JSONDecodeError, KeyError):
            return None

    def _parse_args(self, raw_args: str) -> dict:
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            return {"input": raw_args}

        if isinstance(parsed, dict):
            return parsed

        return {"input": parsed}

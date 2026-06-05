from __future__ import annotations

import json

from react_agent_from_scratch.output_parser import (
    FinalAnswer,
    ParsedToolCall,
    ParserFailure,
    AgentOutputParser,
)


def test_parses_final_answer() -> None:
    parser = AgentOutputParser()

    result = parser.parse(json.dumps({
        "thought": "I know the answer.",
        "type": "final_answer",
        "answer": "MCP is a protocol for connecting AI apps to tools.",
    }))

    assert isinstance(result, FinalAnswer)
    assert result.content == "MCP is a protocol for connecting AI apps to tools."


def test_parses_valid_tool_call() -> None:
    parser = AgentOutputParser()

    result = parser.parse(json.dumps({
        "thought": "I need current information.",
        "type": "tool_call",
        "tool_name": "web_search",
        "args": {"query": "latest MCP Python SDK"},
    }))

    assert isinstance(result, ParsedToolCall)
    assert result.thought == "I need current information."
    assert result.tool_call.name == "web_search"
    assert result.tool_call.args == {"query": "latest MCP Python SDK"}


def test_invalid_format_returns_parser_failure() -> None:
    parser = AgentOutputParser()

    result = parser.parse("I should probably search the web first.")

    assert isinstance(result, ParserFailure)
    assert "not valid JSON" in result.message
    assert result.raw_output == "I should probably search the web first."


def test_invalid_tool_json_returns_parser_failure() -> None:
    parser = AgentOutputParser()

    result = parser.parse("{ this is not valid json }")

    assert isinstance(result, ParserFailure)


def test_missing_tool_name_returns_parser_failure() -> None:
    parser = AgentOutputParser()

    result = parser.parse(json.dumps({
        "thought": "I should search.",
        "type": "tool_call",
        "args": {"query": "MCP SDK"},
    }))

    assert isinstance(result, ParserFailure)
    assert "tool_name" in result.message

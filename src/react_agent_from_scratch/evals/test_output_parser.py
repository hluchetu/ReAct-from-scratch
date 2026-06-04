from __future__ import annotations

from react_agent_from_scratch.output_parser import (
    FinalAnswer,
    ParsedToolCall,
    ParserFailure,
    ReActOutputParser,
)


def test_parses_final_answer() -> None:
    parser = ReActOutputParser()

    result = parser.parse(
        "Final Answer: MCP is a protocol for connecting AI apps to tools."
    )

    assert isinstance(result, FinalAnswer)
    assert result.content == "MCP is a protocol for connecting AI apps to tools."


def test_parses_valid_tool_call() -> None:
    parser = ReActOutputParser()

    result = parser.parse(
        """
        Thought: I need current information.
        Tool Call: {"name": "web_search", "args": {"query": "latest MCP Python SDK"}}
        """
    )

    assert isinstance(result, ParsedToolCall)
    assert result.thought == "I need current information."
    assert result.tool_call.name == "web_search"
    assert result.tool_call.args == {"query": "latest MCP Python SDK"}


def test_invalid_format_returns_parser_failure() -> None:
    parser = ReActOutputParser()

    result = parser.parse("I should probably search the web first.")

    assert isinstance(result, ParserFailure)
    assert "expected ReAct format" in result.message
    assert result.raw_output == "I should probably search the web first."


def test_invalid_tool_json_returns_parser_failure() -> None:
    parser = ReActOutputParser()

    result = parser.parse(
        """
        Thought: I should search.
        Tool Call: {"name": "web_search", "args": {"query": "MCP SDK"}
        """
    )

    assert isinstance(result, ParserFailure)
    assert "Tool call JSON was invalid" in result.message


def test_missing_tool_name_returns_parser_failure() -> None:
    parser = ReActOutputParser()

    result = parser.parse(
        """
        Thought: I should search.
        Tool Call: {"args": {"query": "MCP SDK"}}
        """
    )

    assert isinstance(result, ParserFailure)
    assert "Tool call JSON was invalid" in result.message

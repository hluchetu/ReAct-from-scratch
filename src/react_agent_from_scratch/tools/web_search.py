from __future__ import annotations

from tavily import TavilyClient

from ..tools import Tool, ToolExecutionResult, ToolInputSchema


def _format_results(results: list[dict]) -> str:
    lines = []
    for r in results:
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")
        lines.append(f"**{title}**\n{url}\n{content}")
    return "\n\n".join(lines)


def build_web_search_tool(api_key: str) -> Tool:
    client = TavilyClient(api_key=api_key)

    def run(args: dict) -> ToolExecutionResult:
        query = args.get("query", "")
        if not query:
            return ToolExecutionResult(
                content="Error: 'query' argument is required.",
                status="error",
            )

        response = client.search(
            query=query,
            search_depth="basic",
            max_results=5,
        )

        results = response.get("results", [])
        if not results:
            return ToolExecutionResult(
                content=f"No results found for: {query}",
                status="success",
            )

        return ToolExecutionResult(
            content=_format_results(results),
            metadata={"query": query, "result_count": len(results)},
            status="success",
        )

    return Tool(
        name="web_search",
        description=(
            "Search the web for current information. "
            "Use for recent events, live facts, prices, news, or anything that may have changed. "
            "Input must be a concise keyword query."
        ),
        input_schema=ToolInputSchema(
            properties={"query": {"type": "string", "description": "The search query"}},
            required=["query"],
        ),
        run=run,
    )

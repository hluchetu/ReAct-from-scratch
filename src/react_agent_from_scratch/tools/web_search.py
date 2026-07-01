from __future__ import annotations

from typing import Any

from sophons.tools import FunctionTool, tool


def _format_results(results: list[dict]) -> str:
    lines = []
    for r in results:
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")
        lines.append(f"**{title}**\n{url}\n{content}")
    return "\n\n".join(lines)


def build_sophons_web_search_tool(api_key: str) -> FunctionTool:
    from tavily import TavilyClient

    client = TavilyClient(api_key=api_key)

    @tool
    def web_search(query: str) -> dict[str, Any]:
        """Search the web for current information."""
        if not query:
            raise ValueError("'query' argument is required.")

        response = client.search(
            query=query,
            search_depth="basic",
            max_results=5,
        )
        results = response.get("results", [])

        return {
            "query": query,
            "result_count": len(results),
            "content": _format_results(results)
            if results
            else f"No results found for: {query}",
            "results": results,
        }

    return web_search

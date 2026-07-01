from __future__ import annotations

from typing import Any

from sophons.tools import tool

from react_agent_from_scratch.config import settings


def _format_results(results: list[dict]) -> str:
    lines = []
    for r in results:
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")
        lines.append(f"**{title}**\n{url}\n{content}")
    return "\n\n".join(lines)


def _make_client():
    from tavily import TavilyClient

    return TavilyClient(api_key=settings.tavily_api_key)


_client = _make_client()


@tool
def web_search(query: str) -> dict[str, Any]:
    """Search the web for current information."""
    if not query:
        raise ValueError("'query' argument is required.")

    response = _client.search(
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

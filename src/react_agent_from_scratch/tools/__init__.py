from __future__ import annotations

from .web_search import web_search


def get_all_tools() -> list:
    return [web_search]

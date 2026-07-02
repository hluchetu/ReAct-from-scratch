from __future__ import annotations

from sophons.integrations.tools import tavily_web_search

from react_agent_from_scratch.config import settings

web_search = tavily_web_search(api_key=settings.tavily_api_key)

from __future__ import annotations

from sophons.agents import Agent
from sophons.agents import AgentResult
from sophons.agents.session import InMemorySessionManager

from sophons.integrations.models import ModelSettings

import datetime

from .config import settings
from .llm.provider import ModelProviderRegistry
from .prompts import build_system_prompt
from .tools.web_search import build_sophons_web_search_tool


def build_agent(model_ref: str = "ollama:qwen3:4b") -> Agent:
    registry = ModelProviderRegistry()
    tools = [build_sophons_web_search_tool(api_key=settings.tavily_api_key)]
    provider_name, model_name = ModelProviderRegistry._parse_model_ref(model_ref)
    model = registry.get_model(
        provider_name=provider_name,
        model_name=model_name,
        settings=ModelSettings(),
        tools=tools,
    )

    return Agent(
        model=model,
        tools=tools,
        system_prompt=build_system_prompt(datetime.date.today().isoformat()),
        session_manager=InMemorySessionManager(),
    )


def main() -> None:
    agent = build_agent("ollama:qwen3:4b")
    print("ReAct Agent ready. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue
        result: AgentResult = agent.run_sync(user_input)
        print(f"\nAgent: {result.message}")
        print(
            f"[{result.stop_reason.value} | steps={result.metrics.steps} | "
            f"model_calls={result.metrics.model_calls} | "
            f"tool_calls={result.metrics.tool_calls}]\n"
        )

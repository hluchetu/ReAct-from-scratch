from __future__ import annotations

import datetime

from sophons.agents import Agent, AgentResult
from sophons.agents.session import InMemorySessionManager
from sophons.integrations.models import DeepSeekModel

from .config import settings
from .prompts import build_system_prompt
from .tools import get_all_tools


def init_agent() -> Agent:
    tools = get_all_tools()
    model = DeepSeekModel(
        model="deepseek-chat",
        api_key=settings.deepseek_api_key,
    )

    return Agent(
        model=model,
        tools=tools,
        system_prompt=build_system_prompt(datetime.date.today().isoformat()),
        session_manager=InMemorySessionManager(),
    )


def main() -> None:
    agent = init_agent()
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

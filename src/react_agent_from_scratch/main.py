from __future__ import annotations

from .config import settings
from .llm.agent import ReActAgent
from .llm.model import ModelSettings
from .llm.provider import ModelProviderRegistry
from .observability.provider import TracerProvider
from .prompts import SYSTEM_PROMPT
from .tools import ToolRegistry
from .tools.web_search import build_web_search_tool


def build_agent(model_ref: str = "ollama:qwen3:4b") -> ReActAgent:
    registry = ModelProviderRegistry()
    tracer = TracerProvider().get_tracer()
    provider_name, model_name = ModelProviderRegistry._parse_model_ref(model_ref)
    model = registry.get_model(
        provider_name=provider_name,
        model_name=model_name,
        settings=ModelSettings(),
    )

    tools = ToolRegistry()
    tools.register_tool(build_web_search_tool(api_key=settings.tavily_api_key))

    return ReActAgent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        tracer=tracer,
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
        result = agent.run(user_input)
        print(f"\nAgent: {result.content}")
        print(f"[{result.stop_reason} | steps={result.steps} | model_calls={result.model_calls} | tool_calls={result.tool_calls}]\n")

from __future__ import annotations

from .llm.model import ModelSettings
from .llm.provider import ModelProviderRegistry
from .llm.agent import ReActAgent
from .tools import Tool, ToolExecutionResult, ToolInputSchema, ToolRegistry
from .prompts import SYSTEM_PROMPT


def build_agent(model_ref: str = "ollama:qwen3:4b") -> ReActAgent:
    registry = ModelProviderRegistry()
    provider_name, model_name = ModelProviderRegistry._parse_model_ref(model_ref)
    model = registry.get_model(
        provider_name=provider_name,
        model_name=model_name,
        settings=ModelSettings(),
    )

    tools = ToolRegistry()
    tools.register_tool(
        Tool(
            name="web_search",
            description="Search the web for current information.",
            input_schema=ToolInputSchema(
                properties={"query": {"type": "string"}},
                required=["query"],
            ),
            run=lambda args: ToolExecutionResult(
                content=f"[web_search stub] No results for: {args.get('query')}"
            ),
        )
    )

    return ReActAgent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
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
        response = agent.run(user_input)
        print(f"\nAgent: {response}\n")

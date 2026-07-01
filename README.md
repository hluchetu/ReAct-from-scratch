# ReAct Agent From Scratch

A Python ReAct agent built on the [Sophons](../sophons) SDK. The agent reasons, calls tools, and loops until it has a final answer — with short-term memory across conversation turns.

## How it works

```text
User message
  → model decides to call a tool (or answer directly)
  → Sophons executes the tool
  → result is added to history
  → model is called again with full context
  → final answer returned
```

All provider clients, adapters, and the agent loop live in Sophons. This repo is the application layer: config, tools, prompts, and CLI.

## Project Structure

```text
src/react_agent_from_scratch/
├── config.py          # Pydantic settings loaded from .env
├── main.py            # Builds the agent (model + tools + session)
├── prompts.py         # System prompt (date injected at runtime)
├── cli.py             # Rich interactive chat CLI (react-chat)
├── tools/
│   ├── __init__.py    # get_all_tools() — register tools here
│   └── web_search.py  # Tavily web search tool
└── evals/
    └── test_sophons_agent.py
```

## Setup

Sophons must be checked out beside this repo:

```text
~/sophons             ← Sophons SDK
~/ReAct-from-scratch
```

Install:

```bash
uv sync
```

Configure `.env`:

```env
DEEPSEEK_API_KEY=your_key
TAVILY_API_KEY=your_key
```

## Run

```bash
react-chat
```

## Key Concepts

**Direct model instantiation** — Models are instantiated directly from Sophons: `DeepSeekModel(model="deepseek-reasoner", api_key=...)`. No registry, no string parsing. To switch models, change the model name in `main.py`.

**Chain-of-thought reasoning** — Uses `deepseek-reasoner` (DeepSeek R1), which produces a separate `reasoning_content` field showing its step-by-step thinking before the final answer. The CLI displays this in a magenta "Thinking" panel before each response.

**Tools injected at call time** — Tools are defined once on `Agent` and passed to `model.invoke()` on each call. The model never holds tool state.

**Short-term memory** — The CLI passes the same `session_id` on every turn. Sophons' `InMemorySessionManager` accumulates user/assistant pairs, so follow-up questions like "how about Arsenal?" resolve correctly.

**Date-aware prompting** — Today's date is injected into the system prompt at agent startup so the model includes the correct year in web search queries.

**Web search** — Powered by Tavily. The system prompt requires the agent to always search for sports, news, prices, and anything time-sensitive rather than relying on training data.

## Adding a tool

1. Create `tools/my_tool.py` with a `@tool` decorated function
2. Add it to `get_all_tools()` in `tools/__init__.py`

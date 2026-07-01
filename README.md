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
├── llm/
│   └── provider.py    # Registry: resolves "provider:model" strings
├── tools/
│   └── web_search.py  # Tavily web search tool
└── evals/
    └── test_sophons_agent.py
```

## Supported Providers

Providers are implemented in Sophons and registered here. Switch with `--model`:

| Provider   | Example ref                      | Needs                  |
|------------|----------------------------------|------------------------|
| DeepSeek   | `deepseek:deepseek-chat`         | `DEEPSEEK_API_KEY`     |
| Ollama     | `ollama:qwen3:4b`                | Ollama running locally |
| Anthropic  | `anthropic:claude-sonnet-4-6`    | `ANTHROPIC_API_KEY`    |

Adding a new provider = add one adapter file in Sophons + register it in `provider.py`.

## Setup

Sophons must be checked out beside this repo:

```text
~/sophons          ← Sophons SDK
~/ReAct-from-scratch
```

Install:

```bash
uv sync
```

Configure `.env`:

```env
DEEPSEEK_API_KEY=your_key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
OLLAMA_BASE_URL=http://localhost:11434
ANTHROPIC_API_KEY=your_key
TAVILY_API_KEY=your_key
```

## Run

```bash
react-chat                                    # default: deepseek:deepseek-chat
react-chat --model ollama:qwen3:4b
react-chat --model anthropic:claude-sonnet-4-6
```

## Key Concepts

**Adapter pattern** — Each provider family (OpenAI-compatible, Anthropic) has its own message serializer in Sophons. DeepSeek and Ollama share `OpenAICompatAdapter`. Adding Anthropic only required a new `AnthropicAdapter` — nothing else changed.

**Short-term memory** — The CLI passes the same `session_id` on every turn. Sophons' `InMemorySessionManager` accumulates user/assistant pairs, so follow-up questions like "how about Arsenal?" resolve correctly.

**Date-aware prompting** — Today's date is injected into the system prompt at agent startup so the model includes the correct year in web search queries.

**Web search** — Powered by Tavily. The system prompt requires the agent to always search for sports, news, prices, and anything time-sensitive rather than relying on training data.

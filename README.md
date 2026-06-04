# ReAct Agent From Scratch

A ReAct (Reasoning + Acting) agent built from scratch in Python — no LangChain, no LangGraph. Just pure Python, Pydantic, and direct LLM API calls.

## What is ReAct?

ReAct is a prompting pattern where the agent alternates between:

1. **Thought** — reasoning about what to do next
2. **Action** — calling a tool
3. **Observation** — reading the tool result

...until it has enough information to give a **Final Answer**.

## Project Structure

```
src/react_agent_from_scratch/
├── config.py            # Pydantic settings — loads env variables from .env
├── context.py           # Message types: SystemMessage, UserMessage, AssistantMessage, ToolMessage
├── prompts.py           # System prompt for the ReAct agent
├── tools.py             # Tool, ToolRegistry, ToolExecutionResult
├── output_parser.py     # Parses raw LLM output into FinalAnswer or ParsedToolCall
├── run.py               # RunConfig, RunLimits, RunState, AgentRunResult, StopReason
├── main.py              # Wiring layer — assembles and runs the agent
├── llm/
│   ├── model.py         # Model and ModelProvider protocols
│   ├── llm_client.py    # LLMClient protocol
│   ├── ollama.py        # OllamaModel + OllamaProvider
│   ├── deepseek.py      # DeepSeekClient + DeepSeekProvider
│   ├── provider.py      # ModelProviderRegistry
│   └── agent.py         # ReActAgent — the core loop
└── observability/
    ├── config.py        # Observability settings
    ├── tracing.py       # Tracer protocol and trace event types
    ├── langfuse.py      # Langfuse tracer implementation
    ├── phoenix.py       # Arize Phoenix tracer implementation
    └── provider.py      # TracerProvider — selects tracer from config
```

## Setup

**1. Clone and install dependencies**

```bash
git clone git@github.com:hluchetu/ReAct-from-scratch.git
cd ReAct-from-scratch
uv sync
```

**2. Configure environment variables**

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```env
DEEPSEEK_API_KEY=your_key_here
OLLAMA_BASE_URL=http://localhost:11434
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# Observability (optional)
OBSERVABILITY_ENABLED=true
OBSERVABILITY_PROVIDER=langfuse   # or: phoenix

# Langfuse
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

**3. Run**

```bash
uv run react-agent-from-scratch
```

## Supported LLM Providers

| Provider | Notes |
|---|---|
| Ollama | Local models, no API key needed |
| DeepSeek | Supports extended thinking via `reasoning_effort` |

Providers are resolved by a `provider:model` reference string:

```python
registry.get_model("ollama", "qwen3:4b", settings)
registry.get_model("deepseek", "deepseek-chat", settings)
```

## Key Concepts

**`ModelProvider` / `Model` protocols** — any LLM backend only needs to implement `get_model()` and `generate()`. No inheritance required (structural typing via `Protocol`).

**`ToolRegistry`** — register any callable as a tool with a name, description, and input schema. The agent calls `execute_tool()` which handles errors and returns a `ToolMessage` so the model can self-correct.

**`ReActOutputParser`** — parses the raw LLM string output into either a `ParsedToolCall` (with extracted `thought` and `ToolCall`) or a `FinalAnswer`.

**`ChatContext`** — tracks the full conversation as a list of typed messages and formats them for the model.

**`RunConfig` / `RunState`** — `RunConfig` holds four circuit breakers (`max_steps`, `max_model_calls`, `max_tool_calls`, `max_runtime_seconds`). `RunState` tracks live counters during a run and records why the agent stopped.

**`AgentRunResult`** — structured return type from `agent.run()`. Carries the answer, `stop_reason`, `success` flag, and call counts. Enables callers to branch on outcome without parsing strings.

**Observability** — pluggable tracing via `TracerProvider`. Set `OBSERVABILITY_PROVIDER=langfuse` or `OBSERVABILITY_PROVIDER=phoenix` in `.env` to record every model call, tool execution, and agent step to your observability platform.

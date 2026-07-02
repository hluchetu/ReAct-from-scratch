from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from sophons.agents.hooks import AfterModelCall, HookRegistry

from .main import init_agent

console = Console()

_PROMPT_STYLE = Style.from_dict({"prompt": "bold #5f87ff"})

_HISTORY_FILE = Path.home() / ".react_agent_history"


def _print_header(model_ref: str) -> None:
    console.print()
    console.print(
        Panel.fit(
            f"[bold white]ReAct Agent[/bold white]  [dim]{model_ref}[/dim]\n"
            "[dim]Type your question and press Enter. [bold]exit[/bold] or Ctrl+C to quit.[/dim]",
            border_style="bright_black",
            padding=(0, 2),
        )
    )
    console.print()


def _print_user(text: str) -> None:
    console.print(
        Panel(
            Text(text, style="white"),
            title="[bold #5f87ff]You[/bold #5f87ff]",
            border_style="#5f87ff",
            padding=(0, 1),
        )
    )


def _print_assistant(text: str, metrics) -> None:
    subtitle = (
        f"[dim]steps={metrics.steps}  "
        f"model_calls={metrics.model_calls}  "
        f"tool_calls={metrics.tool_calls}[/dim]"
    )
    console.print(
        Panel(
            Markdown(text),
            title="[bold #2dba4e]Agent[/bold #2dba4e]",
            subtitle=subtitle,
            border_style="#2dba4e",
            padding=(0, 1),
        )
    )


def _print_tool_use(tool_name: str) -> None:
    console.print(f"  [dim]→ using tool:[/dim] [bold yellow]{tool_name}[/bold yellow]")


def _thinking_hook() -> HookRegistry:
    registry = HookRegistry()

    def on_after_model(event: AfterModelCall) -> None:
        reasoning = event.message.metadata.get("reasoning")
        if reasoning:
            console.print(Panel(
                reasoning,
                title="[bold magenta]Thinking[/bold magenta]",
                border_style="magenta",
                padding=(0, 1),
            ))

    registry.add(on_after_model)
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="react-chat", description="ReAct agent chat CLI"
    )
    parser.parse_args()

    _print_header("deepseek-reasoner")

    try:
        agent = init_agent(hooks=_thinking_hook())
    except Exception as exc:
        console.print(f"[bold red]Failed to build agent:[/bold red] {exc}")
        sys.exit(1)

    session_id = agent.new_session_id()

    session: PromptSession = PromptSession(
        history=FileHistory(str(_HISTORY_FILE)),
        style=_PROMPT_STYLE,
    )

    while True:
        try:
            user_input = session.prompt("  You › ", style=_PROMPT_STYLE).strip()
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted.[/dim]")
            break
        except EOFError:
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "/exit", "/quit"}:
            console.print("[dim]Goodbye.[/dim]")
            break

        console.print()
        _print_user(user_input)
        console.print()

        try:
            with console.status("[dim]Thinking...[/dim]", spinner="dots"):
                result = agent.run_sync(user_input, session_id=session_id)

            console.print()
            _print_assistant(result.message, result.metrics)
            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]Cancelled.[/dim]\n")
        except Exception as exc:
            console.print(f"\n[bold red]Error:[/bold red] {exc}\n")

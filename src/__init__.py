import os
import uuid
import warnings

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from src.agent import DEFAULT_MODEL, build_graph, text_of
from .sandbox import Sandbox

console = Console()

SLASH_COMMANDS: dict[str, str] = {
    "/help": "Show available slash commands",
    "/info": "Show current model, sandbox, and thread info",
    "/clear": "Clear conversation memory (new thread)",
    "/reset": "Recreate the Docker sandbox and start a fresh thread",
    "/model": "Show or switch model (usage: /model <provider:name>)",
    "/exec": "Run a shell command directly in the sandbox (usage: /exec <cmd>)",
    "/exit": "Exit Terminator",
}


class SlashCompleter(Completer):
    """Show slash command completions as soon as the user types '/'."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return
        for cmd, desc in SLASH_COMMANDS.items():
            if cmd.startswith(text):
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display=cmd,
                    display_meta=desc,
                )


def show(m) -> None:
    """Render an agent or tool message with Rich formatting."""
    if m.type == "ai":
        for tc in m.tool_calls:
            cmd = tc.get("args", {}).get("cmd")
            if tc["name"] == "run_command" and isinstance(cmd, str):
                body = Syntax(cmd.strip(), "bash", theme="monokai", word_wrap=True)
            else:
                body = Text(str(tc.get("args", {})), style="cyan")
            console.print(
                Panel(
                    body,
                    title=f"[bold yellow]⚙ {tc['name']}[/]",
                    title_align="left",
                    border_style="yellow",
                    padding=(0, 1),
                )
            )

        if text := text_of(m.content):
            console.print(
                Panel(
                    Markdown(text),
                    title="[bold green]Terminator[/]",
                    title_align="left",
                    border_style="green",
                    padding=(0, 1),
                )
            )
    elif m.type == "tool":
        output = (m.content or "").strip() or "(no output)"
        console.print(
            Panel(
                Text(output, style="dim"),
                title="[dim]output[/]",
                title_align="left",
                border_style="bright_black",
                padding=(0, 1),
            )
        )


def _print_help() -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    for cmd, desc in SLASH_COMMANDS.items():
        table.add_row(f"[bold cyan]{cmd}[/]", f"[dim]{desc}[/]")
    console.print(Panel(table, title="[bold]Slash Commands[/]", border_style="cyan", padding=(0, 1)))


def _handle_slash(line: str, ctx: dict) -> bool:
    """Handle a slash command. Returns False if the CLI should exit."""
    parts = line.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "/help":
        _print_help()
    elif cmd == "/exit":
        return False
    elif cmd == "/info":
        cid = getattr(ctx["sandbox"].container, "short_id", "unknown")
        ws = ctx["workspace"] or "(none)"
        console.print(
            Panel(
                f"[bold]Model:[/] [cyan]{ctx['model']}[/]\n"
                f"[bold]Thread:[/] [dim]{ctx['thread_id']}[/]\n"
                f"[bold]Container:[/] [dim]{cid}[/]\n"
                f"[bold]Workspace:[/] [cyan]{ws}[/]",
                title="[bold]Session Info[/]",
                border_style="cyan",
                padding=(0, 1),
            )
        )
    elif cmd == "/clear":
        ctx["thread_id"] = str(uuid.uuid4())
        console.print("[green]Started a fresh conversation thread.[/]")
    elif cmd == "/reset":
        with console.status("[bold cyan]Resetting Docker sandbox...[/]", spinner="dots"):
            ctx["sandbox"].close()
            ctx["sandbox"] = Sandbox(workspace=ctx["workspace"])
            ctx["graph"] = build_graph(ctx["sandbox"], ctx["model"])
            ctx["thread_id"] = str(uuid.uuid4())
        console.print("[green]Sandbox recreated and conversation reset.[/]")
    elif cmd == "/model":
        if not arg:
            console.print(f"[bold]Current model:[/] [cyan]{ctx['model']}[/]")
        else:
            try:
                with console.status(f"[bold cyan]Switching to {arg}...[/]", spinner="dots"):
                    ctx["graph"] = build_graph(ctx["sandbox"], arg)
                    ctx["model"] = arg
                console.print(f"[green]Switched model to[/] [cyan]{arg}[/]")
            except Exception as exc:
                console.print(f"[red]Failed to switch model:[/] {exc}")
    elif cmd == "/exec":
        if not arg:
            console.print("[yellow]Usage:[/] /exec <command>")
        else:
            with console.status("[bold cyan]Running in sandbox...[/]", spinner="dots"):
                out = ctx["sandbox"].execute(arg)
            console.print(
                Panel(
                    Text(out.strip() or "(no output)", style="dim"),
                    title=f"[bold yellow]⚙ exec:[/] [dim]{arg}[/]",
                    title_align="left",
                    border_style="yellow",
                    padding=(0, 1),
                )
            )
    else:
        console.print(f"[red]Unknown command:[/] {cmd}. Type [cyan]/help[/] to see available commands.")

    return True


def main() -> None:
    load_dotenv()
    model_name = os.getenv("TERMINATOR_MODEL") or DEFAULT_MODEL
    workspace = os.getenv("TERMINATOR_WORKSPACE") or None

    with console.status("[bold cyan]Starting Docker sandbox...[/]", spinner="dots"):
        sandbox = Sandbox(workspace=workspace)
        graph = build_graph(sandbox, model_name)

    ctx = {
        "sandbox": sandbox,
        "graph": graph,
        "model": model_name,
        "workspace": workspace,
        "thread_id": str(uuid.uuid4()),
    }

    ws_info = f" | workspace: [cyan]{workspace}[/]" if workspace else ""
    console.print(
        Panel(
            f"[bold]Model:[/] [cyan]{model_name}[/]{ws_info}\n"
            "[dim]Type [bold]/[/] for commands, or [bold]exit[/] / [bold]Ctrl+D[/] to stop.[/]",
            title="[bold red]Terminator[/]",
            border_style="red",
            padding=(0, 1),
        )
    )

    session = PromptSession(
        history=InMemoryHistory(),
        completer=SlashCompleter(),
        complete_while_typing=True,
    )

    try:
        while True:
            try:
                user = session.prompt(HTML("<ansiblue><b>❯</b></ansiblue> ")).strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Shutting down...[/]")
                break

            if not user:
                continue
            if user in ("exit", "quit"):
                break
            if user.startswith("/"):
                if not _handle_slash(user, ctx):
                    break
                continue

            config = {
                "configurable": {"thread_id": ctx["thread_id"]},
                "recursion_limit": 40,
            }
            with console.status("[bold cyan]Thinking...[/]", spinner="dots") as status:
                for state in ctx["graph"].stream(
                    {"messages": [{"role": "user", "content": user}]},
                    config,
                    stream_mode="values",
                ):
                    msg = state["messages"][-1]
                    if msg.type in ("ai", "tool"):
                        status.stop()
                        show(msg)
                        if msg.type == "tool" or (msg.type == "ai" and msg.tool_calls):
                            status.start()
    finally:
        with console.status("[dim]Cleaning up sandbox...[/]", spinner="dots"):
            ctx["sandbox"].close()

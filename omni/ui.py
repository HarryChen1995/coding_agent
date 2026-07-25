"""Rich-powered terminal presentation for the coding agent.

Nothing here affects agent logic — it's purely how things are shown. If rich
isn't installed, agent.py falls back to plain print() (see its import guard).
"""

import json
import os
import re

from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.syntax import Syntax
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console()

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _render_diff(diff_text: str) -> Text:
    """Render a unified diff (as produced by difflib.unified_diff) with a
    line-number gutter and red/green highlighting for removed/added lines —
    GitHub-style — instead of relying on pygments' diff-lexer coloring."""
    body = Text()
    old_no = new_no = None
    for line in diff_text.splitlines():
        if line.startswith("---") or line.startswith("+++"):
            continue
        m = _HUNK_RE.match(line)
        if m:
            old_no, new_no = int(m.group(1)), int(m.group(2))
            body.append(f"{line}\n", style="dim cyan")
            continue
        if line.startswith("-"):
            gutter = old_no if old_no is not None else ""
            body.append(f"{gutter:>5} ", style="dim")
            body.append(f"{line}\n", style="bold red")
            if old_no is not None:
                old_no += 1
        elif line.startswith("+"):
            gutter = new_no if new_no is not None else ""
            body.append(f"{gutter:>5} ", style="dim")
            body.append(f"{line}\n", style="bold green")
            if new_no is not None:
                new_no += 1
        else:
            gutter = new_no if new_no is not None else ""
            body.append(f"{gutter:>5} ", style="dim")
            body.append(f"{line}\n")
            if old_no is not None:
                old_no += 1
            if new_no is not None:
                new_no += 1
    return body


def _search_summary(result: str) -> str:
    """search_files results can be dozens of matched lines — the step log
    should read as a count, not a code dump (the model still gets the full
    text; this only affects what's printed to the terminal)."""
    if result.strip() == "(no matches)":
        return "0 matches found"
    lines = [l for l in result.splitlines() if l and not l.startswith("...")]
    count = len(lines)
    stopped = "...[stopped at" in result
    suffix = " (stopped early — narrow your pattern or glob)" if stopped else ""
    return f"{count} match{'es' if count != 1 else ''} found{suffix}"


def _read_summary(result: str) -> str:
    """read_file's result is the raw file content — the step log should
    say how much was read, not echo the code (the model still gets the
    full text; this only affects what's printed to the terminal)."""
    lines = result.splitlines()
    return f"{len(lines)} line{'s' if len(lines) != 1 else ''} ({len(result)} chars)"

_STATUS_COLOR = {"done": "green", "running": "yellow", "max_steps": "yellow", "error": "red"}

_LEXER_BY_EXT = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "tsx",
    ".jsx": "jsx", ".go": "go", ".rs": "rust", ".java": "java", ".rb": "ruby",
    ".c": "c", ".cpp": "cpp", ".h": "c", ".sh": "bash", ".yaml": "yaml",
    ".yml": "yaml", ".json": "json", ".md": "markdown", ".sql": "sql",
    ".html": "html", ".css": "css",
}


def _lexer_for(path: str) -> str:
    return _LEXER_BY_EXT.get(os.path.splitext(path)[1], "text")


def banner(task: str, model: str):
    console.print(Rule("[bold cyan]Coding Agent[/bold cyan]"))
    console.print(f"[dim]model:[/dim] {model}")
    console.print(f"[dim]task:[/dim]  {task}\n")


def interactive_banner(model: str, resumed: str = None):
    console.print(Rule("[bold cyan]Coding Agent — interactive[/bold cyan]"))
    console.print(f"[dim]model:[/dim] {model}")
    if resumed:
        console.print(f"[dim]resuming session:[/dim] {resumed}")
    console.print("[dim]Type a task, /sessions to list saved sessions, /exit to quit. "
                  "Ctrl+C interrupts the current turn without leaving the session.[/dim]\n")


async def prompt_task_async(session) -> str:
    """Read one line of input via a prompt_toolkit PromptSession, so the
    input line stays pinned to the bottom of the terminal — all other
    output (parsing/thinking spinners, panels, results) scrolls in the
    region above it instead of interleaving with the prompt. `session` is
    a prompt_toolkit.PromptSession the caller creates once and reuses
    across turns (so up-arrow history works). Must be called inside the
    caller's `patch_stdout()` context so Rich's output (which resolves
    sys.stdout lazily on every print) is redrawn above the pinned prompt
    instead of corrupting it."""
    console.print(Rule(style="dim"))
    return await session.prompt_async(HTML("<ansigreen><b>❯</b></ansigreen> "))


def thinking(label: str = "Thinking…"):
    """Spinner shown while waiting on a model call. Safe to use around an
    `await` — rich's status display refreshes on its own thread, so it
    doesn't block the event loop."""
    return console.status(f"[bold cyan]{label}[/bold cyan]", spinner="dots")


def intent_panel(intent, existing: dict):
    risk_color = {"low": "green", "medium": "yellow", "high": "red"}.get(intent.risk_level, "white")
    files_line = "none specified"
    if intent.target_files:
        parts = []
        for f in intent.target_files:
            tag = "[green]exists[/green]" if existing.get(f) else "[yellow]new[/yellow]"
            parts.append(f"{f} ({tag})")
        files_line = ", ".join(parts)
    constraints_line = "; ".join(intent.constraints) if intent.constraints else "none stated"
    confidence = "" if intent.confident else "\n[red]⚠ low confidence — parsing fell back to defaults[/red]"

    body = (
        f"[bold]type:[/bold] {intent.task_type}    "
        f"[bold]risk:[/bold] [{risk_color}]{intent.risk_level}[/{risk_color}]\n"
        f"[bold]summary:[/bold] {intent.summary}\n"
        f"[bold]files:[/bold] {files_line}\n"
        f"[bold]constraints:[/bold] {constraints_line}"
        f"{confidence}"
    )
    console.print(Panel(body, title="Parsed Intent", border_style="blue", expand=False))


def high_risk_warning():
    console.print(Panel(
        "Approval required for ALL write/shell actions this run, even with --auto-approve.",
        title="⚠ High-risk task detected", border_style="red", expand=False,
    ))


_READONLY_TOOLS = {"read_file", "list_dir", "search_files", "glob_files", "git_diff"}
_WRITE_TOOLS = {"write_file", "edit_file"}

_CATEGORY_COLOR = {
    "readonly": "cyan",
    "write": "yellow",
    "shell": "magenta",
    "other": "white",
}


def _category(name: str) -> str:
    if name in _READONLY_TOOLS:
        return "readonly"
    if name in _WRITE_TOOLS:
        return "write"
    if name == "run_shell":
        return "shell"
    return "other"


def _format_args(args: dict, max_len: int = 60) -> str:
    parts = []
    for k, v in args.items():
        if isinstance(v, str):
            shown = v if len(v) <= max_len else v[:max_len] + "…"
            display = json.dumps(shown, ensure_ascii=False)
        else:
            display = repr(v)
        parts.append(f"{k}={display}")
    return ", ".join(parts)


def step_header(step: int, name: str, args: dict):
    color = _CATEGORY_COLOR[_category(name)]
    call_str = f"{name}({_format_args(args)})"
    console.print(f"\n[bold cyan]step {step}[/bold cyan] → [{color}]{call_str}[/{color}]")


async def request_approval(name: str, args: dict, client) -> bool:
    """Show a rich preview (diff / content / command) and ask for confirmation.
    `client` is an MCPToolClient — previews go through the same MCP server
    the real tool calls do, just via the read-only _preview_* tools."""
    if name == "edit_file":
        ok, preview = await client.preview_edit(args.get("path", ""), args.get("old_str", ""), args.get("new_str", ""))
        if not ok:
            console.print(Panel(f"[red]{preview}[/red]", title=f"edit_file: {args.get('path')}", border_style="red"))
            return False
        console.print(Panel(_render_diff(preview),
                             title=f"edit_file: {args.get('path')}", border_style="yellow"))
    elif name == "write_file":
        path = args.get("path", "")
        is_new, preview = await client.preview_write(path, args.get("content", ""), args.get("overwrite", False))
        if is_new:
            console.print(Panel(Syntax(preview, _lexer_for(path), theme="ansi_dark", word_wrap=True),
                                 title=f"write_file (new): {path}", border_style="green"))
        else:
            console.print(Panel(_render_diff(preview),
                                 title=f"write_file (overwrite): {path}", border_style="yellow"))
    elif name == "run_shell":
        cmd = args.get("command", "")
        console.print(Panel(Syntax(cmd, "bash", theme="ansi_dark"),
                             title="run_shell", border_style="magenta"))
    else:
        console.print(Panel(json.dumps(args, indent=2), title=name, border_style="white"))

    return Confirm.ask("[bold]Proceed?[/bold]", default=False)


def tool_result(step: int, name: str, result: str, ok: bool):
    icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
    if ok and name == "search_files":
        summary = _search_summary(result)
    elif ok and name == "read_file":
        summary = _read_summary(result)
    else:
        summary = result.splitlines()[0] if result else ""
    console.print(f"  {icon} {summary[:160]}")
    # For edits, show the diff body under the summary line — with a line
    # number gutter and red/green highlighting for removed/added lines.
    if ok and name == "edit_file" and "\n" in result:
        diff_body = result.split("\n", 1)[1]
        console.print(_render_diff(diff_body))


def final_result(text: str):
    console.print(Rule("[bold green]Done[/bold green]"))
    console.print(Panel(Markdown(text), border_style="green"))


def history_panel(messages: list):
    """Show the prior conversation being resumed, rendered the same way it
    looked the first time around — assistant replies as Markdown panels,
    same as final_result() — so it's visibly clear context carried over
    instead of silently feeding the model in the background."""
    visible = [m for m in messages if m.get("role") != "system"]
    console.print(Rule(f"[bold blue]Resumed history — {len(visible)} messages[/bold blue]"))

    for m in visible:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role == "user":
            console.print(f"\n[bold green]❯[/bold green] {content}")
        elif role == "assistant" and m.get("tool_calls"):
            calls = ", ".join(f"{c['function']['name']}(…)" for c in m["tool_calls"])
            console.print(f"[dim]  → called {calls}[/dim]")
        elif role == "assistant":
            if content:
                console.print(Panel(Markdown(content), border_style="cyan", expand=False))
        elif role == "tool":
            summary = content.splitlines()[0] if content else ""
            console.print(f"  [dim]✓ {summary[:150]}[/dim]")

    console.print(Rule(style="dim"))


def sessions_table(sessions: list):
    table = Table(title="Saved Sessions", expand=False)
    table.add_column("id", style="bold cyan")
    table.add_column("name", style="bold magenta")
    table.add_column("status")
    table.add_column("updated", style="dim")
    table.add_column("model", style="dim")
    table.add_column("task")

    for s in sessions:
        color = _STATUS_COLOR.get(s["status"], "white")
        task = s["task"] if len(s["task"]) <= 60 else s["task"][:60] + "…"
        table.add_row(s["id"], s.get("name") or "-", f"[{color}]{s['status']}[/{color}]",
                      s["updated_at"], s["model"], task)

    console.print(table)


def interrupted():
    console.print(
        "\n[yellow]⏹ Interrupted — back at the prompt. Progress up to the last "
        "completed step was saved; keep chatting or ask the agent to continue.[/yellow]"
    )


def warning(text: str):
    console.print(f"[yellow]⚠ {text}[/yellow]")


def error(text: str):
    console.print(f"[red]✗ {text}[/red]")

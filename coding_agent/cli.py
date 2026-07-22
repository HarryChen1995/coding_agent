"""CLI entry point (Typer).

Examples:
    python cli.py "Add type hints to utils.py and run the tests" \\
        --project-root ./myrepo

    python cli.py "Fix the failing test in test_math.py" \\
        --project-root ./myrepo --auto-approve

    python cli.py --session-name refactor-utils "Add type hints to utils.py"

    python cli.py --list-sessions

    python cli.py --resume refactor-utils "also add a docstring"

    python cli.py --delete-session refactor-utils

    python cli.py                      # no task -> interactive REPL, fresh session
    python cli.py --resume refactor-utils   # no task -> interactive REPL, resumed session

    python cli.py --mcp-server "weather=python -m weather_mcp_server" \\
        --mcp-server "docs=node docs-server.js --port 4000" "Look up today's forecast"

    python cli.py --add-mcp-server "weather=python -m weather_mcp_server"  # register once,
    python cli.py "what's the forecast?"                                   # available from here on, no flags needed

    python cli.py --list-mcp-servers
    python cli.py --remove-mcp-server weather

    python cli.py --help
"""

import asyncio
import os
from typing import List, Optional

import typer

from .agent import CodingAgent
from .config import AgentConfig
from .mcp_client import (
    MCPToolClient, default_mcp_config_path, load_mcp_config,
    parse_mcp_server_specs, save_mcp_config,
)
from .session_store import SessionStore

app = typer.Typer(add_completion=False, help="Coding agent (Ollama + Qwen Coder)")


@app.command()
def main(
    task: Optional[str] = typer.Argument(
        None, help="What you want the agent to do. Optional with --resume (continues "
                   "with no new instruction) or --list-sessions.",
    ),
    project_root: str = typer.Option(".", "--project-root", "-p", help="Directory the agent is scoped to"),
    model: str = typer.Option("qwen3.6:35b", "--model", "-m", help="Ollama model to drive the agent"),
    ollama_host: Optional[str] = typer.Option(
        None, "--ollama-host", help="Ollama server URL (defaults to $OLLAMA_HOST or http://localhost:11434)",
    ),
    ollama_api_key: Optional[str] = typer.Option(
        None, "--ollama-api-key",
        help="Bearer token if Ollama sits behind an authenticated proxy "
             "(defaults to $OLLAMA_API_KEY — prefer the env var over this flag "
             "so the key doesn't end up in your shell history).",
    ),
    max_steps: int = typer.Option(25, "--max-steps", help="Hard cap on agent loop iterations"),
    auto_approve: bool = typer.Option(
        False, "--auto-approve",
        help="Skip human approval for write/edit/shell tools. Only use in an "
             "already-isolated environment (container/VM). Overridden if intent parsing flags the task high-risk.",
    ),
    log_path: str = typer.Option("agent_run.log", "--log-path", help="Where to write the structured run log"),
    skip_intent_parsing: bool = typer.Option(
        False, "--skip-intent-parsing",
        help="Skip the upfront structured-intent parse and go straight into the agent loop.",
    ),
    intent_model: Optional[str] = typer.Option(
        None, "--intent-model", help="Smaller/faster model to use just for intent parsing (defaults to --model)",
    ),
    db_path: str = typer.Option(
        "agent_sessions.db", "--db-path", help="SQLite file storing session/message history",
    ),
    resume: Optional[str] = typer.Option(
        None, "--resume", help="Resume a previous session by id or --session-name instead of starting a new one",
    ),
    session_name: Optional[str] = typer.Option(
        None, "--session-name", help="Give a new session a memorable name, so you can --resume it by name later",
    ),
    list_sessions: bool = typer.Option(
        False, "--list-sessions", help="List saved sessions (id, name, status, task) and exit",
    ),
    delete_session: Optional[str] = typer.Option(
        None, "--delete-session", help="Delete a saved session (by id or --session-name) and exit",
    ),
    mcp_config: Optional[str] = typer.Option(
        None, "--mcp-config",
        help='Path to a Claude-Desktop-style MCP config file ({"mcpServers": {"name": '
             '{"command": ..., "args": [...], "env": {...}}}}) to load extra tools from, '
             "alongside the built-in ones. Their tools appear to the model as <name>__<tool>.",
    ),
    mcp_server: List[str] = typer.Option(
        [], "--mcp-server",
        help='Add one custom MCP server inline, format "name=command arg1 arg2 ...". '
             "Repeatable for multiple servers. Merged with --mcp-config if both are given "
             "(this flag wins on a name clash). Its tools appear to the model as <name>__<tool>.",
    ),
    add_mcp_server: Optional[str] = typer.Option(
        None, "--add-mcp-server",
        help='Register a custom MCP server permanently (format "name=command arg1 arg2 ..."), '
             "then exit. Saved to ~/.coding_agent/mcp.json and auto-loaded on every future run "
             "— no need to pass --mcp-server/--mcp-config again.",
    ),
    remove_mcp_server: Optional[str] = typer.Option(
        None, "--remove-mcp-server", help="Remove a permanently-registered MCP server by name, then exit",
    ),
    list_mcp_servers: bool = typer.Option(
        False, "--list-mcp-servers", help="List permanently-registered MCP servers and exit",
    ),
):
    """Run the coding agent on TASK inside PROJECT_ROOT. Omit TASK to enter
    an interactive session (fresh, or resumed with --resume)."""
    if add_mcp_server:
        try:
            spec = parse_mcp_server_specs([add_mcp_server])
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        path = default_mcp_config_path()
        servers = load_mcp_config(path) if os.path.exists(path) else {}
        servers.update(spec)
        save_mcp_config(path, servers)
        (name,) = spec.keys()
        typer.echo(f"Registered MCP server {name!r} in {path} — available on every run from now on.")
        raise typer.Exit()

    if remove_mcp_server:
        path = default_mcp_config_path()
        servers = load_mcp_config(path) if os.path.exists(path) else {}
        if remove_mcp_server not in servers:
            typer.echo(f"Error: no registered MCP server named {remove_mcp_server!r}.", err=True)
            raise typer.Exit(code=1)
        del servers[remove_mcp_server]
        save_mcp_config(path, servers)
        typer.echo(f"Removed MCP server {remove_mcp_server!r}.")
        raise typer.Exit()

    if list_mcp_servers:
        path = default_mcp_config_path()
        servers = load_mcp_config(path) if os.path.exists(path) else {}
        if not servers:
            typer.echo("No registered MCP servers.")
        else:
            for name, spec in servers.items():
                typer.echo(f"{name}: {spec['command']} {' '.join(spec.get('args', []))}")
        raise typer.Exit()

    if delete_session:
        if SessionStore(db_path).delete_session(delete_session):
            typer.echo(f"Deleted session {delete_session!r}.")
        else:
            typer.echo(f"Error: no session found with id or name {delete_session!r}.", err=True)
            raise typer.Exit(code=1)
        raise typer.Exit()

    if list_sessions:
        _print_sessions(SessionStore(db_path).list_sessions())
        raise typer.Exit()

    try:
        extra_mcp_servers = parse_mcp_server_specs(mcp_server)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    # Explicit --mcp-config wins; otherwise auto-load the global registry
    # (~/.coding_agent/mcp.json) if it exists, so servers added once via
    # --add-mcp-server are available on every run without any flags.
    effective_mcp_config_path = mcp_config or (
        default_mcp_config_path() if os.path.exists(default_mcp_config_path()) else ""
    )

    cfg = AgentConfig(
        model=model,
        ollama_host=ollama_host or "",
        ollama_api_key=ollama_api_key or "",
        project_root=project_root,
        max_steps=max_steps,
        auto_approve=auto_approve,
        log_path=log_path,
        parse_intent=not skip_intent_parsing,
        intent_model=intent_model or "",
        db_path=db_path,
        mcp_config_path=effective_mcp_config_path,
        mcp_servers=extra_mcp_servers,
    )

    if task is None:
        asyncio.run(_interactive(cfg, resume, session_name))
        return

    if resume:
        _show_resumed_history(db_path, resume)

    agent = CodingAgent(cfg)
    try:
        result = asyncio.run(agent.run(task, resume_session_id=resume, session_name=session_name))
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    try:
        from . import ui
        ui.final_result(result)
    except ImportError:
        typer.echo("\n=== FINAL RESULT ===")
        typer.echo(result)


async def _interactive(cfg: AgentConfig, resume: Optional[str], session_name: Optional[str]):
    """REPL: keep one MCP client open across turns (avoids re-spawning the
    tool-server subprocess every turn) and keep resuming the same session
    (fresh on turn 1, then whatever session that turn created/resumed)."""
    agent = CodingAgent(cfg)
    session_id = resume

    try:
        from . import ui
        ui.interactive_banner(cfg.model, resumed=resume)
    except ImportError:
        typer.echo(f"Interactive mode (model: {cfg.model}). Type a task, /sessions to list, /exit to quit.\n")

    if resume:
        _show_resumed_history(cfg.db_path, resume)

    async with MCPToolClient(cfg.project_root, mcp_config_path=cfg.mcp_config_path or None,
                              extra_servers=cfg.mcp_servers or None) as client:
        while True:
            try:
                task = _read_task()
            except (EOFError, KeyboardInterrupt):
                typer.echo()
                break

            task = task.strip()
            if not task:
                continue
            if task in ("/exit", "/quit"):
                break
            if task == "/sessions":
                _print_sessions(agent.store.list_sessions())
                continue
            if task.startswith("/delete "):
                target = task[len("/delete "):].strip()
                if agent.store.delete_session(target):
                    typer.echo(f"Deleted session {target!r}.")
                    if session_id is not None and agent.store.resolve_session_id(session_id) is None:
                        session_id = None  # the session we were resuming just got deleted
                else:
                    typer.echo(f"No session found with id or name {target!r}.", err=True)
                continue

            try:
                result = await agent.run(task, resume_session_id=session_id, client=client,
                                          session_name=session_name, show_banner=False)
            except ValueError as e:
                typer.echo(f"Error: {e}", err=True)
                continue

            session_id = agent.session_id

            try:
                from . import ui
                ui.final_result(result)
            except ImportError:
                typer.echo("\n=== RESULT ===")
                typer.echo(result)


def _show_resumed_history(db_path: str, resume: str):
    """Print the conversation being resumed so it's visible on screen that
    context actually carried over — agent.run() feeds it to the model
    either way, but nothing else displays it."""
    store = SessionStore(db_path)
    session_id = store.resolve_session_id(resume)
    if session_id is None:
        return  # let agent.run() raise the proper "no session found" error
    messages = store.load_messages(session_id)
    try:
        from . import ui
        ui.history_panel(messages)
    except ImportError:
        typer.echo(f"--- Resumed history ({len(messages)} messages) ---")
        for m in messages:
            if m.get("role") == "system":
                continue
            typer.echo(f"{m.get('role')}: {str(m.get('content'))[:200]}")
        typer.echo("--- end history ---\n")


def _read_task() -> str:
    try:
        from . import ui
        return ui.prompt_task()
    except ImportError:
        return input("> ")


def _print_sessions(sessions: list):
    if not sessions:
        typer.echo("No saved sessions.")
        return
    try:
        from . import ui
        ui.sessions_table(sessions)
    except ImportError:
        for s in sessions:
            typer.echo(f"{s['id']}  {s.get('name') or '-'}  [{s['status']}]  {s['updated_at']}  {s['task'][:70]}")


if __name__ == "__main__":
    app()

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

    python cli.py --add-mcp-server "docs=node docs-server.js" --defer  # tools loaded on demand
    python cli.py --mcp-server "docs=node docs-server.js,defer" "..."  # same, one-off via suffix

    python cli.py --list-mcp-servers
    python cli.py --remove-mcp-server weather

    python cli.py --help
"""

import asyncio
import os
import shlex
import signal
from contextlib import nullcontext
from typing import List, Optional

import typer

from .agent import CodingAgent
from .config import AgentConfig
from .llm_client import LLMError, list_models
from .mcp_client import (
    MCPToolClient, default_mcp_config_path, load_mcp_config,
    parse_mcp_server_specs, save_mcp_config,
)
from .session_store import SessionStore

_STATIC_COMMANDS = {
    "/exit": "leave the REPL",
    "/quit": "leave the REPL",
    "/sessions": "list saved sessions",
    "/delete ": "delete a saved session — /delete <id-or-name>",
    "/compact": "summarize this session's history down to a briefing",
    "/model": "list models available on the LLM server (also populates /model <name> below)",
}

app = typer.Typer(add_completion=False, help="Coding agent (Qwen Coder or any OpenAI-compatible model)")


@app.command()
def main(
    task: Optional[str] = typer.Argument(
        None, help="What you want the agent to do. Optional with --resume (continues "
                   "with no new instruction) or --list-sessions.",
    ),
    project_root: str = typer.Option(".", "--project-root", "-p", help="Directory the agent is scoped to"),
    model: str = typer.Option("qwen3.6:35b", "--model", "-m", help="Model name to drive the agent"),
    llm_host: Optional[str] = typer.Option(
        None, "--llm-host", help="OpenAI-compatible server URL (defaults to $LLM_HOST or http://localhost:11434)",
    ),
    llm_api_key: Optional[str] = typer.Option(
        None, "--llm-api-key",
        help="Bearer token if the LLM server sits behind an authenticated proxy "
             "(defaults to $LLM_API_KEY — prefer the env var over this flag "
             "so the key doesn't end up in your shell history).",
    ),
    max_steps: int = typer.Option(100, "--max-steps", help="Hard cap on agent loop iterations"),
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
    embedding_model: Optional[str] = typer.Option(
        None, "--embedding-model",
        help="Embedding backend for search_tools semantic ranking against deferred MCP tool "
             'descriptions. Defaults to "nomic-local" — on-device via `pip install "nomic[local]"`, '
             "no server needed. Pass a remote OpenAI-compatible embedding model name (e.g. mxbai-embed-large) "
             'to use that instead, or "" to disable and fall back to plain keyword matching.',
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
             "(this flag wins on a name clash). Its tools appear to the model as <name>__<tool>. "
             'Append ",defer" (e.g. "name=command args...,defer") to keep this server\'s tools '
             "out of the model's default tool list — it discovers them on demand via search_tools.",
    ),
    add_mcp_server: Optional[str] = typer.Option(
        None, "--add-mcp-server",
        help='Register a custom MCP server permanently (format "name=command arg1 arg2 ..."), '
             "then exit. Saved to ~/.omni-coder/mcp.json and auto-loaded on every future run "
             "— no need to pass --mcp-server/--mcp-config again.",
    ),
    defer: bool = typer.Option(
        False, "--defer",
        help="With --add-mcp-server: don't expose this server's tools to the model up front. "
             "Instead a search_tools tool is offered; the model calls it with a query to load "
             "matching tools on demand, keeping unused tool schemas out of context. Default: false "
             '(equivalent to appending ",defer" to the --add-mcp-server / --mcp-server spec).',
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
        (name,) = spec.keys()
        if defer:
            spec[name]["defer"] = True
        path = default_mcp_config_path()
        servers = load_mcp_config(path) if os.path.exists(path) else {}
        servers.update(spec)
        save_mcp_config(path, servers)
        suffix = " (deferred tool loading)" if spec[name].get("defer") else ""
        typer.echo(f"Registered MCP server {name!r} in {path}{suffix} — available on every run from now on.")
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
                target = spec["url"] if "url" in spec else f"{spec['command']} {' '.join(spec.get('args', []))}"
                suffix = " [defer]" if spec.get("defer") else ""
                typer.echo(f"{name}: {target}{suffix}")
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
    # (~/.omni-coder/mcp.json) if it exists, so servers added once via
    # --add-mcp-server are available on every run without any flags.
    effective_mcp_config_path = mcp_config or (
        default_mcp_config_path() if os.path.exists(default_mcp_config_path()) else ""
    )

    cfg = AgentConfig(
        model=model,
        llm_host=llm_host or "",
        llm_api_key=llm_api_key or "",
        project_root=project_root,
        max_steps=max_steps,
        auto_approve=auto_approve,
        log_path=log_path,
        parse_intent=not skip_intent_parsing,
        intent_model=intent_model or "",
        db_path=db_path,
        mcp_config_path=effective_mcp_config_path,
        mcp_servers=extra_mcp_servers,
        # None (flag omitted) -> AgentConfig's own default ("nomic-embed-text");
        # "" (--embedding-model "" explicitly) -> disabled.
        embedding_model=embedding_model if embedding_model is not None else AgentConfig.embedding_model,
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
    (fresh on turn 1, then whatever session that turn created/resumed).

    Input is read through a prompt_toolkit PromptSession wrapped in
    patch_stdout(), so the input line stays pinned to the bottom of the
    terminal — parsing/thinking spinners, panels, and results all scroll in
    the region above it instead of interleaving with the prompt. Falls back
    to a plain input() loop if rich/prompt_toolkit aren't installed."""
    agent = CodingAgent(cfg)
    session_id = resume
    commands = dict(_STATIC_COMMANDS)  # mutated in place below once MCP prompts are discovered

    try:
        from . import ui
        from prompt_toolkit import PromptSession
        from prompt_toolkit.patch_stdout import patch_stdout
        ui.header(cfg.model, f"{resume} (resumed)" if resume else "(new)")
        prompt_session = PromptSession(completer=ui.SlashCommandCompleter(commands), complete_while_typing=True)
        # raw=True: pass Rich's ANSI-coded output straight through instead of
        # patch_stdout()'s default write() path, which sanitizes/escapes text
        # (it assumes plain text) and mangles embedded escape codes into
        # literal garbage like "?[32m" on the screen.
        stdout_cm = patch_stdout(raw=True)
    except ImportError:
        typer.echo(f"Interactive mode (model: {cfg.model}). Type a task, /sessions to list, "
                   "/compact to summarize a long session's history, /model to list/switch models, "
                   "/exit to quit. Ctrl+C interrupts the current turn without leaving the session.\n")
        prompt_session = None
        stdout_cm = nullcontext()

    if resume:
        _show_resumed_history(cfg.db_path, resume)

    async with MCPToolClient(cfg.project_root, mcp_config_path=cfg.mcp_config_path or None,
                              extra_servers=cfg.mcp_servers or None,
                              embedding_model=cfg.embedding_model or None,
                              llm_host=cfg.llm_host or None,
                              llm_api_key=cfg.llm_api_key or None) as client:
        prompts = await client.list_prompts()
        for name, info in prompts.items():
            arg_hint = " ".join(
                f"<{a['name']}>" if a["required"] else f"[{a['name']}]" for a in info["arguments"]
            )
            commands[f"/{name} "] = f"{info['description']} {arg_hint}".strip()

        try:
            # Best-effort: some LLM servers don't expose /v1/models. Register
            # each model name as its own "/model <name>" completion so typing
            # "/model " pops a pickable list — /model (bare) below refreshes
            # this same set, in case models changed since startup.
            for m in await list_models(cfg.llm_host or None, cfg.llm_api_key or None):
                commands[f"/model {m}"] = "switch to this model"
        except LLMError:
            pass

        with stdout_cm:
            while True:
                try:
                    task = await _read_task(prompt_session)
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
                if task == "/compact":
                    if session_id is None:
                        typer.echo("No active session yet — run a task first.")
                    else:
                        typer.echo("Compacting history…")
                        typer.echo(await agent.compact_history(session_id))
                    continue
                if task == "/model":
                    try:
                        models = await list_models(cfg.llm_host or None, cfg.llm_api_key or None)
                    except LLMError as e:
                        typer.echo(f"Error: {e}", err=True)
                        continue
                    for m in models:
                        commands[f"/model {m}"] = "switch to this model"

                    if prompt_session is None or not models:
                        # No prompt_toolkit (plain input() fallback), or the
                        # server returned no models: fall back to a static
                        # list — pick with "/model <name>" instead.
                        typer.echo(f"Current model: {cfg.model}")
                        for m in models:
                            typer.echo(f"  {'* ' if m == cfg.model else '  '}{m}")
                        continue

                    from prompt_toolkit.shortcuts import radiolist_dialog
                    selected = await radiolist_dialog(
                        title="Select model",
                        text=f"Current: {cfg.model}  (↑/↓ to move, Enter to select, Esc to cancel)",
                        values=[(m, m) for m in models],
                        default=cfg.model if cfg.model in models else None,
                    ).run_async()
                    if selected and selected != cfg.model:
                        cfg.model = selected
                        typer.echo(f"Switched to model {cfg.model!r}.")
                        _print_header(cfg, session_id or "(new)")
                    continue
                if task.startswith("/model "):
                    cfg.model = task[len("/model "):].strip()
                    typer.echo(f"Switched to model {cfg.model!r}.")
                    _print_header(cfg, session_id or "(new)")
                    continue
                if task.startswith("/"):
                    prompt_name, _, rest = task[1:].partition(" ")
                    if prompt_name in prompts:
                        arg_specs = prompts[prompt_name]["arguments"]
                        try:
                            values = shlex.split(rest)
                        except ValueError as e:
                            typer.echo(f"Error parsing arguments: {e}", err=True)
                            continue
                        if len(values) > len(arg_specs):
                            names = ", ".join(a["name"] for a in arg_specs) or "(none)"
                            typer.echo(
                                f"Error: /{prompt_name} takes at most {len(arg_specs)} "
                                f"argument(s): {names}", err=True,
                            )
                            continue
                        # MCP prompt arguments are string-typed (dict[str, str]) — shlex.split
                        # already yields plain strings, so no coercion is needed here.
                        prompt_args = {a["name"]: v for a, v in zip(arg_specs, values)}
                        missing = [a["name"] for a in arg_specs if a["required"] and a["name"] not in prompt_args]
                        if missing:
                            typer.echo(f"Error: /{prompt_name} missing required argument(s): "
                                       f"{', '.join(missing)}", err=True)
                            continue
                        try:
                            task = await client.get_prompt(prompt_name, prompt_args)
                        except Exception as e:
                            typer.echo(f"Error resolving prompt {prompt_name!r}: {e}", err=True)
                            continue
                        typer.echo(f"--- resolved /{prompt_name} ---\n{task}\n")

                # Run the turn as a Task so Ctrl+C can cancel just this turn
                # (via the SIGINT handler below) instead of killing the whole
                # REPL — a raw KeyboardInterrupt raised inside asyncio's own
                # blocking wait can otherwise escape uncaught past this loop
                # entirely. task.cancel() injects CancelledError at the
                # coroutine's next await point (model call, tool call, etc.),
                # unwinding just that turn; the MCP client and session history
                # already written to disk are untouched, so the REPL keeps going.
                run_task = asyncio.ensure_future(
                    agent.run(task, resume_session_id=session_id, client=client,
                              session_name=session_name, show_banner=False)
                )
                previous_sigint = signal.signal(signal.SIGINT, lambda *_: run_task.cancel())
                try:
                    result = await run_task
                except asyncio.CancelledError:
                    session_id = agent.session_id or session_id
                    try:
                        from . import ui
                        ui.interrupted()
                    except ImportError:
                        typer.echo("\n[Interrupted — back to prompt. You can keep chatting in this session.]")
                    continue
                except ValueError as e:
                    typer.echo(f"Error: {e}", err=True)
                    continue
                finally:
                    signal.signal(signal.SIGINT, previous_sigint)

                if agent.session_id != session_id:
                    session_id = agent.session_id
                    _print_header(cfg, session_id)

                try:
                    from . import ui
                    ui.final_result(result)
                except ImportError:
                    typer.echo("\n=== RESULT ===")
                    typer.echo(result)


def _print_header(cfg: AgentConfig, session_label: str):
    """Re-print the header box — used at REPL startup and again whenever
    the model or session identity changes (a /model switch, or the session
    getting a real id after its first turn), so the box on screen never
    goes stale."""
    try:
        from . import ui
        ui.header(cfg.model, session_label)
    except ImportError:
        typer.echo(f"[model: {cfg.model}] [session: {session_label}]")


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


async def _read_task(prompt_session) -> str:
    if prompt_session is not None:
        from . import ui
        return await ui.prompt_task_async(prompt_session)
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

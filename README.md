# Local Coding Agent (Ollama + Qwen Coder)

## Setup
```bash
ollama pull qwen3-coder:30b
pip install -r requirements.txt
```
No `ollama` python package required — the agent talks to an OpenAI-compatible
chat-completions endpoint (`/api/v1/chat/completions`) directly over HTTP via
`httpx`. This works against Ollama itself or a gateway in front of it (e.g.
Open WebUI) — point `--ollama-host` at whichever one you're running.

## Run
```bash
python cli.py "Add type hints to utils.py, then run the test suite" \
    --project-root ./myrepo
```
Add `--auto-approve` to skip confirmation prompts (only in an already-sandboxed
environment, e.g. a container you're fine getting wiped). Run `python cli.py --help`
for the full option list — it's a Typer app, so `--help` is auto-generated and
kept in sync with the code.

Omit the task string to drop into an interactive session instead of a
one-shot run — see [Session management](#session-management) below.

## What makes this "production grade" vs. the first draft

| Concern | First draft | This version |
|---|---|---|
| Editing existing files | Only full overwrite via `write_file` | `edit_file` does exact unique-match replace + shows a unified diff, `write_file` refuses to clobber existing files |
| Path safety | None — agent could read/write anywhere | Every path resolved and checked against `project_root`; escapes raise `SandboxError` |
| Shell safety | Ran anything, unbounded | Denylist for destructive patterns (`rm -rf /`, `sudo`, fork bombs, etc.), timeout, output truncation |
| Human oversight | None | Write/edit/shell calls pause for approval unless the tool is in `safe_tools` or `auto_approve=True` |
| Model reliability | Assumed clean tool-call JSON | Retries with backoff on API errors; malformed tool-call args are caught and reported back to the model instead of crashing |
| Context window | Unbounded growth | Char-budget trimming keeps the running conversation under `context_char_budget` |
| Observability | `print()` only | Structured log file (`agent_run.log`) recording every model call, tool call, args, and result |
| Config | Hardcoded constants | `AgentConfig` dataclass — one place to tune model, sandbox root, limits, policy |
| Sessions | Each run started from a blank conversation | Every message is persisted to SQLite (`session_store.py`); resume by id or name, or run interactively |
| Codebase search | `grep` piped through a subprocess | Pure-Python `search_files` (regex + glob filter, skips `.git`/`node_modules`/etc.) and a `glob_files` tool for pattern-based file discovery |

## Still recommended before real production use

1. **Run it in a container**, not on your host. The path sandbox and shell
   denylist reduce risk but are not a substitute for OS-level isolation —
   treat `run_shell` as "can execute arbitrary code" and contain the blast
   radius accordingly (Docker, gVisor, a disposable VM).
2. **Version control everything.** Require the project root to be a git repo
   and commit before each run, so any agent change is a reviewable diff you
   can revert.
3. **Rate/step limits per user** if this is exposed to a team, not just you.
4. **Swap the char-based context trimming for a real tokenizer** if you hit
   context issues in practice — it's a rough approximation.
5. **Add tests for the tools module** (`tools.py`) in your CI — the sandbox
   check is the one thing you really don't want to regress silently.
6. Qwen3-Coder's native tool-calling is solid but not perfect at this size —
   watch the log for `BAD ARGS` entries; if they're frequent, consider a
   larger quant or `qwen2.5-coder:32b` (dense, less agentic-tuned but very
   reliable on straightforward edits).

## Intent parsing

Before the agent takes any action, the raw task string is parsed by the model
(in strict JSON mode, no tools) into structured intent:

```json
{
  "task_type": "bugfix",
  "summary": "Fix add() which subtracts instead of adding",
  "target_files": ["math_utils.py"],
  "constraints": [],
  "risk_level": "low"
}
```

This gets injected into the conversation as a system message (with each
target file tagged `exists` or `new` in the sandbox), so the model starts
with grounded structure instead of just the raw sentence. Two things follow
from this automatically:

- **High-risk tasks force approval**, even if you ran with `--auto-approve`.
  Detected via `risk_level: "high"` (deletion, deploys, migrations, etc.).
- **Malformed or failed parsing degrades gracefully** — after retries, it
  falls back to `task_type: "other"` with `confident=False` logged, and the
  agent still runs on the raw task text rather than blocking.

Skip it with `--skip-intent-parsing` if you want lower latency on simple
tasks, or point it at a smaller/faster model with `--intent-model`.

## Session management

Every message in the conversation — system, user, assistant, tool results —
is persisted to a SQLite file (`agent_sessions.db` by default, `--db-path` to
change it) as the run happens, via `session_store.py`. A session is done the
moment it's created; nothing extra to opt into.

**Resume a previous run:**
```bash
python cli.py "Add type hints to utils.py" --session-name utils-typing
# ...later, in the same or a different terminal...
python cli.py --resume utils-typing "Also add docstrings"
```
`--resume` accepts either the session id it printed at the end of a run, or
the `--session-name` you gave it. `--session-name` is optional — without it
you just get an 8-character id. When you resume, the prior conversation is
printed before the run continues (assistant replies rendered the same
Markdown-panel way they looked the first time), so it's visibly clear that
context carried over rather than just trusting it happened in the background.

**Browse saved sessions:**
```bash
python cli.py --list-sessions
```
Shows id, name, status (`running` / `done` / `max_steps` / `error`), last
updated time, model, and the original task for each session.

**Delete a session:**
```bash
python cli.py --delete-session utils-typing
```
Removes the session and its full message history. Also available as
`/delete <id-or-name>` from inside interactive mode.

**Interactive mode** — omit the task argument entirely to get a REPL instead
of a one-shot run:
```bash
python cli.py --project-root ./myrepo              # fresh session, prompts for input
python cli.py --resume utils-typing                 # resumes and prompts for input
```
Type a task and press enter to run it; the conversation (and the MCP tool
connection) stays alive between turns, so follow-ups don't pay the cost of
re-parsing intent or re-spawning the tool server. Special inputs:
- `/sessions` — list saved sessions without leaving the REPL
- `/delete <id-or-name>` — delete a saved session without leaving the REPL
- `/exit` or `/quit` (or Ctrl-D / Ctrl-C) — leave

A spinner shows while waiting on the model (initial intent parsing and every
turn), including a live retry counter if a call fails transiently and gets
retried — so a slow or cold-loading model doesn't look like it's hung.

## Terminal UI

`ui.py` renders everything through [rich](https://github.com/Textualize/rich):
banner + parsed intent as a panel, each step with a colored ✓/✗, `edit_file`
diffs and `write_file` new-file content syntax-highlighted by extension,
approval prompts that show the actual diff/command *before* you approve —
not just the raw args — and the final response rendered as Markdown (headers,
lists, code blocks) rather than literal text.

If `rich` isn't installed, `agent.py` and `cli.py` both detect the missing
import and fall back to plain `print()` — nothing breaks, it just looks
like the original CLI.

## Tools as an MCP server

The sandboxed tools now live behind a real MCP server (`mcp_server.py`),
not inline in the agent. The agent is an MCP *client* — it spawns the
server as a subprocess (stdio transport) scoped to `--project-root`, fetches
the tool list, converts it to Ollama's function-calling schema, and calls
tools through the MCP session instead of Python function calls directly.

```
cli.py → agent.py → mcp_client.py ⇄ (stdio) ⇄ mcp_server.py → tools.py
```

What this buys you:
- **Any MCP client can use the same tools** — Claude Desktop, another agent
  framework, a different model entirely — all sharing the identical
  sandbox, denylist, and diff-preview logic in `tools.py`.
- **The server is independently runnable and testable**:
  ```bash
  AGENT_PROJECT_ROOT=/path/to/repo python mcp_server.py
  ```
- Internal tools (`_preview_edit`, `_preview_write`, `_file_exists`) are
  underscore-prefixed and filtered out of what's shown to the LLM in
  `list_llm_tools()` — the agent still calls them directly for approval
  previews and intent validation, the model never sees them.

The agent loop is now `async` end-to-end (an MCP session requires it);
`cli.py` runs it via `asyncio.run()`. Nothing about the approval flow,
retries, logging, or intent parsing changed — they just go through
`MCPToolClient` now instead of a local `Tools` instance.

## Files
- `config.py` — all tunables in one dataclass
- `intent.py` — parses the freeform task into structured intent (task_type, target_files, constraints, risk_level)
- `tools.py` — sandboxed tool implementations (used by `mcp_server.py`, not called directly by the agent anymore)
- `mcp_server.py` — MCP server exposing those tools over stdio
- `mcp_client.py` — async MCP client the agent uses to reach the server
- `ollama_client.py` — raw `httpx` client for the model's OpenAI-compatible chat-completions endpoint — no `ollama` package dependency
- `session_store.py` — SQLite persistence for sessions and their full message history (resume/list/interactive mode)
- `ui.py` — rich terminal rendering (diffs, panels, approval prompts, session tables) — purely presentational
- `agent.py` — the loop: parse intent, call model, approve, execute via MCP, persist, repeat
- `cli.py` — command-line entry point (Typer — `python cli.py --help` for auto-generated, always-in-sync docs)

Point at a non-default Ollama host with `--ollama-host http://some-host:11434`
or the `OLLAMA_HOST` env var (checked in that order).

If your Ollama endpoint sits behind an authenticated proxy, set the key via
environment variable rather than the CLI flag — it avoids the token landing
in your shell history:
```bash
export OLLAMA_API_KEY="sk-..."
python cli.py "task" --project-root ./repo --ollama-host http://your-host:8080
```

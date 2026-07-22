# Ollama Coding Agent

An AI coding agent that plans, edits, and tests code by driving Qwen Coder
(or any Ollama-compatible model) through a scoped set of file and shell
tools, with human approval on every write, edit, or shell command.

## Install

```bash
pip install ollama-coding-agent
ollama pull qwen3-coder:30b
```

## Usage

```bash
coding-agent "Add type hints to utils.py, then run the test suite" \
    --project-root ./myrepo
```

Equivalent: `python -m coding_agent "..." --project-root ./myrepo`.

Omit the task string to enter an interactive session instead:

```bash
coding-agent --project-root ./myrepo
```

Run `coding-agent --help` for the full option list.

## Features

- **Structured intent parsing** — the raw task is classified (bug fix,
  feature, refactor, risk level, target files) before any action is taken,
  and high-risk tasks force human approval even under `--auto-approve`.
- **Session persistence** — every message is saved to SQLite as the run
  happens. Resume a previous run by id or a name you gave it
  (`--resume`), browse saved sessions (`--list-sessions`), or delete one
  (`--delete-session`).
- **Interactive mode** — drop into a REPL that keeps the model connection
  and tool session alive across turns.
- **Human-in-the-loop approval** — every write, edit, or shell command
  shows a diff or command preview before you confirm, unless explicitly
  marked safe or run with `--auto-approve`.
- **Retry and recovery** — transient model failures retry with backoff;
  malformed tool-call output is caught and reported back to the model
  instead of crashing the run.
- **Codebase exploration tools** — regex content search with glob
  filtering, pattern-based file discovery, directory listing, and git diff,
  all skipping noise directories (`.git`, `node_modules`, build output).
- **Extensible via custom MCP servers** — point at any MCP server, local
  (stdio) or remote (SSE / Streamable HTTP), and its tools merge into the
  model's toolset automatically, no code changes required. Register one
  permanently (`--add-mcp-server`, available on every future run) or add
  one per run (`--mcp-server`/`--mcp-config`).

## Architecture

Tools are served over the Model Context Protocol (MCP), not called
in-process — the agent is an MCP *client* that talks to a tool server over
stdio:

```
┌───────────┐     ┌────────────┐     ┌──────────────┐
│  CLI/REPL │ --> │ agent loop │ --> │ intent parser│
└───────────┘     │ (call model│     │ (task ->     │
                  │  approve,  │     │  structured  │
                  │  execute,  │     │  intent)     │
                  │  persist)  │     └──────────────┘
                  └─────┬──────┘
       ┌────────────────┼─────────────────┐
       ▼                ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────────┐
│ model client  │ │  MCP client   │ │  session store    │
│ (chat calls)  │ │ (MCP session) │ │  (SQLite history) │
└───────────────┘ └───────┬───────┘ └───────────────────┘
                          │ stdio subprocess
                          ▼
                   ┌───────────────┐
                   │  MCP server   │
                   └───────┬───────┘
                           ▼
                   ┌───────────────┐
                   │     tools     │
                   │ (read/write/  │
                   │ edit/search/  │
                   │    shell)     │
                   └───────────────┘
```

Because tools are exposed over MCP, any MCP-compatible client — Claude
Desktop, another agent framework, a different model entirely — can reach
the exact same toolset, approval-preview logic, and path scoping. The
reverse also holds: any additional MCP server — local (stdio) or remote
(SSE / Streamable HTTP) — can be plugged into this agent, and its tools
merge into the same list the model already sees —
```bash
coding-agent --add-mcp-server "weather=python -m weather_mcp_server"     # local, stdio
coding-agent --add-mcp-server "weather=https://example.com/mcp/sse"      # remote, SSE
coding-agent "what's the forecast?"   # picked up automatically, every run from here on
```
A value after `name=` starting with `http://`/`https://` is treated as a
remote server (SSE by default, append `,streamable_http` for that transport
instead); anything else is a local command spawned over stdio — it doesn't
need to be `-m`-invokable, a standalone script's absolute path works too
(e.g. `"myserver=python C:/absolute/path/to/mcp_server.py"`).

## Configuration

Point at any Ollama-compatible host with `--ollama-host` or the
`OLLAMA_HOST` env var. If it sits behind an authenticated proxy, set
`OLLAMA_API_KEY` as an environment variable rather than a CLI flag so the
key doesn't end up in shell history.

## Links

Source, full documentation, and issue tracker:
https://github.com/HarryChen1995/coding_agent

## License

MIT

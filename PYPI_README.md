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
  and tool session alive across turns. Ctrl-C during a running turn cancels
  just that turn instead of killing the session — you land back at the
  prompt and can keep going.
- **Human-in-the-loop approval** — every write, edit, or shell command
  shows a diff or command preview before you confirm (diffs render with
  line numbers and red/green highlighting), unless explicitly marked safe
  or run with `--auto-approve`.
- **Retry and recovery** — transient model failures retry with backoff;
  malformed tool-call output is caught and reported back to the model
  instead of crashing the run.
- **Codebase exploration tools** — regex content search with glob
  filtering, pattern-based file discovery, directory listing, and a full
  git toolset (status/log/diff/show/branch/fetch read-only; add/commit/
  pull/push approval-gated), all skipping noise directories (`.git`,
  `node_modules`, build output).
- **Persistent project memory** — the agent can save durable notes (a
  `save_memory` tool call) to a per-project `agent_memory.md`, auto-loaded
  into the system prompt at the start of every new session.
- **Extensible via custom MCP servers** — point at any MCP server, local
  (stdio) or remote (SSE / Streamable HTTP), and its tools merge into the
  model's toolset automatically, no code changes required. Register one
  permanently (`--add-mcp-server`, available on every future run) or add
  one per run (`--mcp-server`/`--mcp-config`).
- **Deferred tool loading + semantic search_tools** — register a custom MCP
  server with `--defer` and its tools stay out of the model's context until
  a synthesized `search_tools` tool loads matching ones on demand, ranked by
  on-device embeddings (`nomic-local`, default) or an Ollama-hosted
  embedding model, with automatic keyword-match fallback.

## Architecture

Tools are served over the Model Context Protocol (MCP), not called
in-process — the agent is an MCP *client* that talks to a tool server over
stdio:

```
+-----------------------------+
|          CLI / REPL         |
+-----------------------------+
               |
               v
+-----------------------------+
|          Agent loop         |
|  parse intent, call model,  |
|  approve, execute, persist  |
+-----------------------------+
               |
               v
+-----------------------------+
|          MCP client         |
|  built-in + custom servers  |
|  merged into one tool list. |
| "defer"-registered servers  |
| hold tools back for on-     |
| demand search_tools lookup  |
+-----------------------------+
               |
 stdio / SSE / streamable-http
               v
+-----------------------------+
|        MCP server(s)        |
+-----------------------------+
               |
               v
+-----------------------------+
|            Tools            |
|    read / write / edit /    |
|        search / shell       |
+-----------------------------+
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

Append `,defer` (or pass `--defer` with `--add-mcp-server`) to keep a
server's tools out of the model's default tool list — it discovers them on
demand via `search_tools`, ranked semantically by default
(`pip install "nomic[local]"` for on-device embeddings, or point
`--embedding-model` at an Ollama-hosted one; `--embedding-model ""` falls
back to plain keyword matching). See the full README for details.

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

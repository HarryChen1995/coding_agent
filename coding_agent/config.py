"""Configuration for the coding agent."""

from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    model: str = "qwen3.6:35b"
    ollama_host: str = ""   # empty = use OLLAMA_HOST env var or http://localhost:11434
    ollama_api_key: str = ""  # empty = use OLLAMA_API_KEY env var; never hardcode this

    # All file/shell operations are confined to this directory. Paths that
    # resolve outside it are rejected before any tool runs.
    project_root: str = "."

    # Tool names that execute WITHOUT asking for human approval first.
    # Anything not listed here (write_file, edit_file, run_shell by default)
    # will print what it's about to do and wait for confirmation, unless
    # auto_approve=True.
    safe_tools: tuple = ("read_file", "list_dir", "search_files", "glob_files", "git_diff")

    auto_approve: bool = False        # True = never prompt (use in CI with care)
    max_steps: int = 25               # hard cap on agent loop iterations

    # Parse the freeform task into structured intent (task_type, target_files,
    # constraints, risk_level) before the agent starts acting.
    parse_intent: bool = True
    intent_model: str = ""            # empty = reuse `model` for intent parsing too
    max_retries: int = 3              # retries per model call on bad/malformed output
    shell_timeout_s: int = 30
    max_output_chars: int = 8000      # truncate tool output before feeding back to model
    context_char_budget: int = 200_000  # rough trim threshold (chars, not tokens)
    log_path: str = "agent_run.log"
    db_path: str = "agent_sessions.db"  # SQLite file storing session/message history

    # Optional path to a Claude-Desktop-style MCP config file
    # ({"mcpServers": {"name": {"command": ..., "args": [...], "env": {...}}}})
    # for adding extra tool servers beyond the built-in one. Empty = none.
    mcp_config_path: str = ""

    # Extra MCP servers specified directly (e.g. via repeatable --mcp-server
    # CLI flags), as {name: {"command": ..., "args": [...], "env": {...}}}.
    # Merged with mcp_config_path's servers; wins on a name clash.
    mcp_servers: dict = field(default_factory=dict)

    # Commands the agent is never allowed to run, regardless of approval.
    denied_shell_patterns: tuple = field(default_factory=lambda: (
        "rm -rf /", "rm -rf /*", ":(){ :|:& };:", "mkfs", "dd if=",
        "> /dev/sda", "shutdown", "reboot", "sudo ", "curl | sh", "wget | sh",
    ))

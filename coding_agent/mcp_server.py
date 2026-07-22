"""MCP server exposing the sandboxed coding tools.

Run standalone to test with any MCP client:
    AGENT_PROJECT_ROOT=/path/to/repo python mcp_server.py

Any MCP-compatible client (not just this agent) can now use these tools —
Claude Desktop, another agent framework, etc. — all sharing the same
sandbox/approval-preview logic in tools.py.
"""

import os

from mcp.server.fastmcp import FastMCP

from .config import AgentConfig
from .tools import Tools

PROJECT_ROOT = os.environ.get("AGENT_PROJECT_ROOT", ".")

cfg = AgentConfig(project_root=PROJECT_ROOT)
impl = Tools(cfg)

mcp = FastMCP("coding-agent-tools")


# ---- Tools exposed to any MCP client (these are what the LLM sees) ----

@mcp.tool()
def read_file(path: str, start_line: int = None, end_line: int = None) -> str:
    """Read a file, optionally a specific line range."""
    return impl.read_file(path, start_line, end_line)


@mcp.tool()
def list_dir(path: str = ".") -> str:
    """List files in a directory."""
    return impl.list_dir(path)


@mcp.tool()
def search_files(pattern: str, path: str = ".", glob: str = None, case_insensitive: bool = False) -> str:
    """Regex search across files under a path, skipping noise directories
    (.git, node_modules, __pycache__, build output, etc.). Optionally
    restrict to filenames matching `glob` (e.g. "*.py")."""
    return impl.search_files(pattern, path, glob, case_insensitive)


@mcp.tool()
def glob_files(pattern: str, path: str = ".") -> str:
    """Find files by glob pattern, e.g. "**/*.tsx" or "src/**/test_*.py".
    Results are sorted newest-first. Use this to discover files by name/
    location; use search_files to find files by content."""
    return impl.glob_files(pattern, path)


@mcp.tool()
def git_diff(path: str = ".") -> str:
    """Show uncommitted git changes in the project."""
    return impl.git_diff(path)


@mcp.tool()
def write_file(path: str, content: str, overwrite: bool = False) -> str:
    """Create a NEW file with content. Fails if the file already exists
    unless overwrite=true. Use edit_file for existing files."""
    return impl.write_file(path, content, overwrite)


@mcp.tool()
def edit_file(path: str, old_str: str, new_str: str) -> str:
    """Replace an exact, unique block of text in an existing file. old_str
    must match precisely (include enough context to be unique)."""
    return impl.edit_file(path, old_str, new_str)


@mcp.tool()
def run_shell(command: str) -> str:
    """Run a shell command in the project root. Dangerous commands are
    blocked by policy."""
    return impl.run_shell(command)


# ---- Internal tools: dry-run previews for the approval UI + existence
# checks for intent validation. Named with a leading underscore so the
# client can filter them out of what it hands to the LLM, while still
# calling them directly for its own approval-flow logic. ----

@mcp.tool()
def _preview_edit(path: str, old_str: str, new_str: str) -> str:
    """Internal: dry-run diff preview for edit_file, no write performed."""
    ok, msg = impl.preview_edit(path, old_str, new_str)
    return f"OK\n{msg}" if ok else f"ERROR\n{msg}"


@mcp.tool()
def _preview_write(path: str, content: str, overwrite: bool = False) -> str:
    """Internal: dry-run preview for write_file, no write performed."""
    is_new, preview = impl.preview_write(path, content, overwrite)
    return f"{'NEW' if is_new else 'DIFF'}\n{preview}"


@mcp.tool()
def _file_exists(path: str) -> str:
    """Internal: sandboxed existence check for intent validation."""
    return "true" if impl.file_exists(path) else "false"


if __name__ == "__main__":
    mcp.run()  # stdio transport by default

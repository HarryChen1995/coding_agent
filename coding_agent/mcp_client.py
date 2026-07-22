"""Thin async MCP client: spawns the built-in tool server as a subprocess
scoped to a project root, optionally alongside any number of additional
("custom") MCP servers a user configures — local (stdio subprocess) or
remote (SSE / Streamable HTTP) — so the agent's tool set isn't limited to
what ships in this package, and new tools can be added without touching
this codebase at all.

Tool schemas from every connected server are merged into one list for the
model. Built-in tools keep their plain names (`read_file`, `write_file`,
...); tools from a custom server are namespaced as `<server_name>__<tool>`
so they can't collide with the built-ins or with each other.
"""

import json
import os
import shlex
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client

_BUILTIN = "_builtin"
_TRANSPORTS = ("sse", "streamable_http")


def default_mcp_config_path() -> str:
    """Global, cross-session MCP server registry — ~/.coding_agent/mcp.json.
    Servers registered here (via `--add-mcp-server`) are available on every
    future run automatically, without passing --mcp-config/--mcp-server."""
    return os.path.join(os.path.expanduser("~"), ".coding_agent", "mcp.json")


def save_mcp_config(path: str, servers: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"mcpServers": servers}, f, indent=2)
        f.write("\n")


def parse_mcp_server_specs(specs: list) -> dict:
    """Parse repeatable `--mcp-server` CLI values into the same shape
    load_mcp_config() returns, so both sources can be merged uniformly.

    Two forms:
      - stdio (local subprocess):  "name=command arg1 arg2 ..."
      - remote (SSE / Streamable HTTP): "name=http://host/mcp/sse" or
        "name=http://host/mcp,streamable_http" (comma + transport to pick
        Streamable HTTP instead of the default SSE)
    """
    servers = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f'Invalid --mcp-server value {spec!r} — expected "name=command args..."')
        name, rest = spec.split("=", 1)
        name, rest = name.strip(), rest.strip()
        if not name:
            raise ValueError(f'Invalid --mcp-server value {spec!r} — expected "name=command args..."')

        if rest.startswith("http://") or rest.startswith("https://"):
            if "," in rest:
                url, transport = (p.strip() for p in rest.rsplit(",", 1))
            else:
                url, transport = rest, "sse"
            if transport not in _TRANSPORTS:
                raise ValueError(f"Invalid --mcp-server transport {transport!r} for {name!r} — expected one of {_TRANSPORTS}")
            servers[name] = {"url": url, "transport": transport}
        else:
            parts = shlex.split(rest)
            if not parts:
                raise ValueError(f'Invalid --mcp-server value {spec!r} — expected "name=command args..."')
            servers[name] = {"command": parts[0], "args": parts[1:]}
    return servers


def load_mcp_config(path: str) -> dict:
    """Read a Claude-Desktop-style MCP config file:

        {
          "mcpServers": {
            "local-server": {
              "command": "python",
              "args": ["-m", "my_tools_server"],
              "env": {"SOME_VAR": "value"}
            },
            "remote-server": {
              "url": "https://example.com/mcp/sse",
              "transport": "sse",
              "headers": {"Authorization": "Bearer ..."}
            }
          }
        }

    Returns the `mcpServers` mapping, or raises ValueError if the file can't
    be read/parsed, or a server entry has neither `command` (stdio) nor
    `url` (sse/streamable_http), or specifies an unknown `transport`.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Could not read MCP config {path!r}: {e}") from e

    servers = data.get("mcpServers", {})
    for name, spec in servers.items():
        if "url" in spec:
            transport = spec.get("transport", "sse")
            if transport not in _TRANSPORTS:
                raise ValueError(f"MCP server {name!r} in {path!r} has unknown transport {transport!r} — expected one of {_TRANSPORTS}")
        elif "command" not in spec:
            raise ValueError(f"MCP server {name!r} in {path!r} must have either 'command' (stdio) or 'url' (sse/streamable_http)")
    return servers


def _mcp_schema_to_ollama(tool, exposed_name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": exposed_name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


class MCPToolClient:
    """Use as an async context manager, one instance per agent run:

        async with MCPToolClient(project_root, mcp_config_path="mcp.json",
                                  extra_servers={"toy": {"command": "python", "args": [...]}}) as client:
            schemas = await client.list_llm_tools()
            result = await client.call_tool("edit_file", {...})

    `mcp_config_path` and `extra_servers` can be used together — servers
    from both are started; on a name collision, `extra_servers` wins.
    """

    def __init__(self, project_root: str, server_path: str = None,
                 mcp_config_path: str = None, extra_servers: dict = None):
        self.project_root = project_root
        # Default: run the built-in server as `python -m <package>.mcp_server`
        # rather than by file path — mcp_server.py uses relative imports
        # (it's part of this package), which only resolve when it's launched
        # as a module, not executed as a standalone script. `server_path` is
        # an escape hatch for pointing the *built-in* slot at a different
        # server entirely.
        self.server_args = [server_path] if server_path else ["-m", f"{__package__}.mcp_server"]
        self.mcp_config_path = mcp_config_path
        self.extra_servers = extra_servers or {}
        self._stack = AsyncExitStack()
        self._sessions: dict = {}     # server name -> ClientSession
        self._tool_owner: dict = {}   # exposed tool name -> (server name, real tool name)

    async def _connect(self, name: str, spec: dict) -> ClientSession:
        try:
            if "url" in spec:
                transport = spec.get("transport", "sse")
                headers = spec.get("headers")
                if transport == "streamable_http":
                    read, write, _ = await self._stack.enter_async_context(
                        streamablehttp_client(spec["url"], headers=headers)
                    )
                else:
                    read, write = await self._stack.enter_async_context(
                        sse_client(spec["url"], headers=headers)
                    )
            else:
                params = StdioServerParameters(
                    command=spec["command"], args=spec.get("args", []),
                    env={**os.environ, **(spec.get("env") or {})},
                )
                read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            return session
        except Exception as e:
            desc = spec.get("url") or f"{spec.get('command')} {' '.join(spec.get('args', []))}"
            raise RuntimeError(f"Failed to start MCP server {name!r} ({desc}): {e}") from e

    async def __aenter__(self):
        builtin_spec = {
            "command": sys.executable, "args": self.server_args,
            "env": {"AGENT_PROJECT_ROOT": self.project_root},
        }
        self._sessions[_BUILTIN] = await self._connect(_BUILTIN, builtin_spec)
        servers = dict(load_mcp_config(self.mcp_config_path)) if self.mcp_config_path else {}
        servers.update(self.extra_servers)  # CLI-specified --mcp-server entries win on name clash
        for name, spec in servers.items():
            self._sessions[name] = await self._connect(name, spec)
        return self

    async def __aexit__(self, *exc_info):
        await self._stack.aclose()

    async def list_llm_tools(self) -> list:
        """Schemas for tools the LLM is allowed to call, merged across every
        connected server. Internal underscore-prefixed built-in tools
        (previews, existence checks) are held back — the agent still calls
        those directly for its own logic. Tools from custom servers are
        namespaced as `<server_name>__<tool_name>`."""
        schemas = []
        self._tool_owner = {}
        for server_name, session in self._sessions.items():
            result = await session.list_tools()
            for tool in result.tools:
                if server_name == _BUILTIN:
                    if tool.name.startswith("_"):
                        continue
                    exposed_name = tool.name
                else:
                    exposed_name = f"{server_name}__{tool.name}"
                self._tool_owner[exposed_name] = (server_name, tool.name)
                schemas.append(_mcp_schema_to_ollama(tool, exposed_name))
        return schemas

    async def call_tool(self, name: str, args: dict) -> str:
        # Internal built-in tools (_preview_edit, etc.) are never registered
        # in _tool_owner since list_llm_tools() filters them out — the
        # (server, name) default below routes them straight to the built-in
        # session under their real name.
        server_name, real_name = self._tool_owner.get(name, (_BUILTIN, name))
        session = self._sessions[server_name]
        result = await session.call_tool(real_name, args)
        text = "".join(c.text for c in result.content if hasattr(c, "text"))
        return f"ERROR: {text}" if result.isError else text

    async def preview_edit(self, path: str, old_str: str, new_str: str):
        raw = await self.call_tool("_preview_edit", {"path": path, "old_str": old_str, "new_str": new_str})
        ok = raw.startswith("OK\n")
        msg = raw.split("\n", 1)[1] if "\n" in raw else raw
        return ok, msg

    async def preview_write(self, path: str, content: str, overwrite: bool = False):
        raw = await self.call_tool("_preview_write", {"path": path, "content": content, "overwrite": overwrite})
        is_new = raw.startswith("NEW\n")
        preview = raw.split("\n", 1)[1] if "\n" in raw else raw
        return is_new, preview

    async def file_exists(self, path: str) -> bool:
        raw = await self.call_tool("_file_exists", {"path": path})
        return raw.strip() == "true"

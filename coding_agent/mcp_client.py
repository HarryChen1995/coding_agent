"""Thin async MCP client: spawns mcp_server.py as a subprocess scoped to a
project root, and adapts its tool list into the Ollama function-calling
schema format the agent already knows how to use.

This is the only file that changed how the agent talks to its tools —
everything else (approval flow, logging, the loop shape) is unchanged;
it just goes through an MCP session now instead of direct Python calls.
"""

import os
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def _mcp_schema_to_ollama(tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


class MCPToolClient:
    """Use as an async context manager, one instance per agent run:

        async with MCPToolClient(project_root) as client:
            schemas = await client.list_llm_tools()
            result = await client.call_tool("edit_file", {...})
    """

    def __init__(self, project_root: str, server_path: str = None):
        self.project_root = project_root
        # Default: run the server as `python -m <package>.mcp_server` rather
        # than by file path — mcp_server.py uses relative imports (it's part
        # of this package), which only resolve when it's launched as a
        # module, not executed as a standalone script. `server_path` is an
        # escape hatch for pointing at a different server entirely.
        self.server_args = [server_path] if server_path else ["-m", f"{__package__}.mcp_server"]
        self._stack = AsyncExitStack()
        self.session: ClientSession = None

    async def __aenter__(self):
        params = StdioServerParameters(
            command=sys.executable,
            args=self.server_args,
            env={**os.environ, "AGENT_PROJECT_ROOT": self.project_root},
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        return self

    async def __aexit__(self, *exc_info):
        await self._stack.aclose()

    async def list_llm_tools(self) -> list:
        """Schemas for tools the LLM is allowed to call — internal
        underscore-prefixed tools (previews, existence checks) are held
        back; the agent still calls those directly for its own logic."""
        result = await self.session.list_tools()
        return [_mcp_schema_to_ollama(t) for t in result.tools if not t.name.startswith("_")]

    async def call_tool(self, name: str, args: dict) -> str:
        result = await self.session.call_tool(name, args)
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

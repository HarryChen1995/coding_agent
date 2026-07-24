"""Agent loop: model <-> MCP tool server, with the guardrails a first draft skips.

Tools now live in mcp_server.py and are reached through mcp_client.MCPToolClient
rather than being called directly — so the loop itself is async (an MCP
session is async under the hood).
"""

import asyncio
import json
import logging
import os
import re
from contextlib import nullcontext

from .ollama_client import chat, OllamaError

from .config import AgentConfig
from .intent import extract_intent
from .mcp_client import MCPToolClient
from .session_store import SessionStore

try:
    from . import ui
    _HAS_UI = True
except ImportError:
    _HAS_UI = False

SYSTEM_PROMPT = """You are a coding agent working within a defined project \
directory. You have tools to read, search, write, and edit files, check git \
diffs, and run shell commands.

Rules:
- Prefer edit_file over write_file for existing files — write_file will \
refuse to overwrite unless you pass overwrite=true.
- Before running anything destructive or irreversible, check git_diff or \
read_file first so you understand current state.
- Keep changes minimal and focused on the task.
- When you learn a durable fact about this project (a convention, gotcha, \
build quirk, or stated preference) that would help in a future session, \
call save_memory to persist it. Keep notes short and skip anything already \
obvious from the code itself.
- When the task is fully done, reply with plain text (no tool call) \
summarizing what changed and how to verify it (e.g. which command to run).
"""


def _load_project_memory(project_root: str, memory_path: str) -> str:
    """Read back whatever save_memory has accumulated for this project, so
    it can be folded into the system prompt at the start of a new session."""
    path = os.path.join(project_root, memory_path)
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("coding_agent")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)
    return logger


async def _approve(tool_name: str, args: dict, cfg: AgentConfig, client: MCPToolClient, force_approval: bool = False) -> bool:
    if tool_name in cfg.safe_tools:
        return True
    if cfg.auto_approve and not force_approval:
        return True
    if _HAS_UI:
        return await ui.request_approval(tool_name, args, client)
    print(f"\n--- Approval needed: {tool_name} ---")
    print(json.dumps(args, indent=2)[:2000])
    answer = input("Proceed? [y/N] ").strip().lower()
    return answer == "y"


def _find_json_objects(text: str) -> list:
    """Scan text for top-level {...} objects, tracking string-literal state
    so braces inside quoted content (e.g. code the model is trying to write)
    don't throw off the balance count."""
    objs = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == "{":
            depth, in_str, esc, j = 0, False, False, i
            while j < n:
                c = text[j]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                elif c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        objs.append(text[i:j + 1])
                        break
                j += 1
            i = j + 1
        else:
            i += 1
    return objs


def _recover_text_tool_calls(content: str, tool_names: set) -> list:
    """Some models print a tool call as plain-text JSON (`{"name": ...,
    "arguments": {...}}`) instead of using the tool-calling API, which would
    otherwise look like a final answer and silently end the run without the
    tool ever executing. Recover any such calls from `content`."""
    if not content or "{" not in content:
        return []
    calls = []
    for raw in _find_json_objects(content):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        name, args = obj.get("name"), obj.get("arguments")
        if name in tool_names and isinstance(args, dict):
            calls.append({
                "id": f"fallback_{len(calls)}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            })
    return calls


def _trim_history(messages: list, budget: int) -> list:
    """Keep the system + user task message plus the most recent turns
    within a rough character budget. Crude but effective without pulling
    in a tokenizer dependency."""
    total = sum(len(str(m.get("content", ""))) for m in messages)
    if total <= budget:
        return messages
    head, tail = messages[:2], messages[2:]
    while tail and total > budget:
        removed = tail.pop(0)
        total -= len(str(removed.get("content", "")))
    return head + tail


class CodingAgent:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.logger = _setup_logger(cfg.log_path)
        self.force_approval = False  # set True for the run if intent is high-risk
        self.store = SessionStore(cfg.db_path)
        self.session_id = None  # set by run() to whichever session the last turn used

    async def _call_model(self, messages: list, tool_schemas: list):
        """Call Ollama with retries for transient errors (connection refused,
        5xx, malformed tool-call output — small local models occasionally
        emit broken JSON)."""
        last_err = None
        spinner = ui.thinking() if _HAS_UI else nullcontext()
        with spinner:
            for attempt in range(1, self.cfg.max_retries + 1):
                try:
                    return await chat(model=self.cfg.model, messages=messages, tools=tool_schemas,
                                       base_url=self.cfg.ollama_host, api_key=self.cfg.ollama_api_key)
                except OllamaError as e:
                    last_err = e
                    self.logger.info(f"model call failed (attempt {attempt}): {e}")
                    if _HAS_UI:
                        spinner.update(f"[bold yellow]Thinking… (retry {attempt}/{self.cfg.max_retries})[/bold yellow]")
                    await asyncio.sleep(min(2 ** attempt, 10))
                except Exception as e:
                    last_err = e
                    self.logger.info(f"unexpected error (attempt {attempt}): {e}")
                    if _HAS_UI:
                        spinner.update(f"[bold yellow]Thinking… (retry {attempt}/{self.cfg.max_retries})[/bold yellow]")
                    await asyncio.sleep(1)
        raise RuntimeError(f"Model call failed after {self.cfg.max_retries} attempts: {last_err}")

    async def run(self, task: str = "", resume_session_id: str = None, client: MCPToolClient = None,
                  session_name: str = None, show_banner: bool = True) -> str:
        """Run one turn of the agent loop. If `client` is given (an already
        -open MCPToolClient), it's reused instead of spawning a fresh MCP
        server subprocess — used by the interactive REPL so each turn
        doesn't pay subprocess-startup cost. `self.session_id` is set to
        whichever session this turn ran against, so callers (e.g. the REPL)
        can pass it back in as `resume_session_id` on the next turn.
        `resume_session_id` accepts either a session id or a --session-name.
        `session_name` optionally names a newly-created session. Pass
        `show_banner=False` when a caller (e.g. the REPL) already prints its
        own header and doesn't want one repeated every turn."""
        resuming = resume_session_id is not None

        if resuming:
            session_id = self.store.resolve_session_id(resume_session_id)
            if session_id is None:
                raise ValueError(f"No session found with id or name {resume_session_id!r}")
            messages = self.store.load_messages(session_id)
            persisted = len(messages)  # already in the DB, don't re-write these
            label = task or "[continuing previous task]"
            if _HAS_UI and show_banner:
                ui.banner(f"(resumed {session_id}) {label}", self.cfg.model)
            self.logger.info(f"RESUME session={session_id} TASK: {label}")
            if task:
                messages.append({"role": "user", "content": task})
        else:
            if _HAS_UI and show_banner:
                ui.banner(task, self.cfg.model)
            session_id = self.store.create_session(self.cfg.project_root, self.cfg.model, task, name=session_name)
            system_content = SYSTEM_PROMPT
            memory_text = _load_project_memory(self.cfg.project_root, self.cfg.memory_path)
            if memory_text:
                system_content += "\n\n# Project memory (persisted from previous sessions)\n" + memory_text
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": task},
            ]
            persisted = 0
            self.logger.info(f"TASK: {task} (session={session_id})")

        self.session_id = session_id

        try:
            if client is not None:
                return await self._run_loop(task, session_id, messages, persisted, resuming, client)
            async with MCPToolClient(self.cfg.project_root, mcp_config_path=self.cfg.mcp_config_path or None,
                                      extra_servers=self.cfg.mcp_servers or None) as owned_client:
                return await self._run_loop(task, session_id, messages, persisted, resuming, owned_client)
        except Exception as e:
            self.store.finish_session(session_id, "error", str(e))
            raise

    async def _run_loop(self, task: str, session_id: str, messages: list, persisted: int,
                         resuming: bool, client: MCPToolClient) -> str:
        tool_schemas = await client.list_llm_tools()
        tool_names = {t["function"]["name"] for t in tool_schemas}

        if not resuming and self.cfg.parse_intent:
            intent_model = self.cfg.intent_model or self.cfg.model
            spinner = ui.thinking("Parsing intent…") if _HAS_UI else nullcontext()
            with spinner:
                intent = await extract_intent(task, intent_model, self.cfg.max_retries, self.logger,
                                               base_url=self.cfg.ollama_host, api_key=self.cfg.ollama_api_key)

            existing = {f: await client.file_exists(f) for f in intent.target_files}
            context_block = intent.as_context_block(existing)
            messages.insert(1, {"role": "system", "content": context_block})

            if _HAS_UI:
                ui.intent_panel(intent, existing)
            else:
                print(f"\n{context_block}\n")

            if intent.risk_level == "high":
                self.force_approval = True
                warning = "High-risk intent detected — approval required for all write/shell actions this run, even with --auto-approve."
                if _HAS_UI:
                    ui.high_risk_warning()
                else:
                    print(f"⚠️  {warning}")
                self.logger.info(warning)

        for m in messages[persisted:]:
            self.store.append_message(session_id, persisted, m)
            persisted += 1

        for step in range(1, self.cfg.max_steps + 1):
            messages = _trim_history(messages, self.cfg.context_char_budget)
            msg = await self._call_model(messages, tool_schemas)

            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                recovered = _recover_text_tool_calls(msg.get("content", ""), tool_names)
                if recovered:
                    msg["tool_calls"] = recovered
                    tool_calls = recovered
                    self.logger.info(
                        f"[step {step}] model printed tool call as plain text; "
                        f"recovered {len(recovered)} call(s) via fallback parsing"
                    )

            messages.append(msg)
            self.store.append_message(session_id, persisted, msg)
            persisted += 1

            if not tool_calls:
                final = msg.get("content", "")
                self.logger.info(f"DONE: {final}")
                self.store.finish_session(session_id, "done", final)
                return final

            for call in tool_calls:
                name = call["function"]["name"]
                args = call["function"]["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        result = f"ERROR: model sent malformed arguments: {args!r}"
                        messages.append({"role": "tool", "content": result})
                        self.store.append_message(session_id, persisted, messages[-1])
                        persisted += 1
                        self.logger.info(f"[step {step}] {name} -> BAD ARGS")
                        continue

                if _HAS_UI:
                    ui.step_header(step, name, args)
                else:
                    print(f"\nstep {step} -> {name}({args})")

                if not await _approve(name, args, self.cfg, client, self.force_approval):
                    result = "Denied by human reviewer. Choose a different approach."
                else:
                    try:
                        result = await client.call_tool(name, args)
                    except Exception as e:
                        result = f"ERROR: {name} raised: {e}"

                ok = not str(result).startswith("ERROR") and result != "Denied by human reviewer. Choose a different approach."
                if _HAS_UI:
                    ui.tool_result(step, name, str(result), ok)
                else:
                    print(f"[step {step}] {name}({args}) -> {str(result)[:200]}")
                self.logger.info(f"[step {step}] {name}({args}) -> {str(result)[:500]}")
                messages.append({"role": "tool", "content": str(result)})
                self.store.append_message(session_id, persisted, messages[-1])
                persisted += 1

        msg = "Max steps reached without completion. Check the log for progress."
        self.logger.info(msg)
        self.store.finish_session(session_id, "max_steps", msg)
        return msg

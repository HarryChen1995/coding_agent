"""Tool implementations. Every tool that touches disk or a shell goes
through the project-scope path check first.
"""

import difflib
import fnmatch
import glob as globmod
import os
import re
import subprocess

from .config import AgentConfig

_IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".idea", ".vscode", "target", ".mypy_cache",
    ".pytest_cache", ".tox", ".ruff_cache",
}


class PathScopeError(Exception):
    pass


def _resolve_in_scope(root: str, path: str) -> str:
    """Resolve `path` relative to `root` and reject any escape attempt
    (../, absolute paths outside root, symlink tricks)."""
    root_abs = os.path.realpath(root)
    candidate = path if os.path.isabs(path) else os.path.join(root_abs, path)
    candidate_abs = os.path.realpath(candidate)
    if not (candidate_abs == root_abs or candidate_abs.startswith(root_abs + os.sep)):
        raise PathScopeError(
            f"Path '{path}' resolves outside the project root ({root_abs}) — denied."
        )
    return candidate_abs


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated, {len(text) - limit} more chars]"


class Tools:
    """Bound to a single AgentConfig so every call is scoped to the project
    root and logged the same way."""

    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg

    # ---- read-only tools (safe by default) ----

    def read_file(self, path: str, start_line: int = None, end_line: int = None) -> str:
        p = _resolve_in_scope(self.cfg.project_root, path)
        if not os.path.isfile(p):
            return f"ERROR: {path} does not exist"
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if start_line or end_line:
            start = (start_line or 1) - 1
            end = end_line or len(lines)
            lines = lines[start:end]
        return _truncate("".join(lines), self.cfg.max_output_chars)

    def file_exists(self, path: str) -> bool:
        """Existence check confined to the project root — used to validate
        parsed intent (e.g. flag target files that don't exist yet) without
        paying the cost of a full read_file call."""
        try:
            p = _resolve_in_scope(self.cfg.project_root, path)
            return os.path.isfile(p)
        except PathScopeError:
            return False

    def list_dir(self, path: str = ".") -> str:
        p = _resolve_in_scope(self.cfg.project_root, path)
        if not os.path.isdir(p):
            return f"ERROR: {path} is not a directory"
        entries = sorted(os.listdir(p))
        return "\n".join(entries) if entries else "(empty)"

    def search_files(self, pattern: str, path: str = ".", glob: str = None,
                      case_insensitive: bool = False) -> str:
        """Regex search across files under `path`, skipping noise directories
        (.git, node_modules, __pycache__, build output, etc.). `glob`
        optionally restricts which filenames are searched (e.g. "*.py")."""
        p = _resolve_in_scope(self.cfg.project_root, path)
        try:
            regex = re.compile(pattern, re.IGNORECASE if case_insensitive else 0)
        except re.error as e:
            return f"ERROR: invalid regex pattern: {e}"

        max_matches = 500
        results = []
        for dirpath, dirnames, filenames in os.walk(p):
            dirnames[:] = [d for d in dirnames if d not in _IGNORE_DIRS]
            for fname in sorted(filenames):
                if glob and not fnmatch.fnmatch(fname, glob):
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        for lineno, line in enumerate(f, start=1):
                            if regex.search(line):
                                rel = os.path.relpath(fpath, self.cfg.project_root)
                                results.append(f"{rel}:{lineno}:{line.rstrip()}")
                                if len(results) >= max_matches:
                                    break
                except OSError:
                    continue
                if len(results) >= max_matches:
                    break
            if len(results) >= max_matches:
                break

        if not results:
            return "(no matches)"
        out = "\n".join(results)
        if len(results) >= max_matches:
            out += f"\n...[stopped at {max_matches} matches — narrow your pattern or glob]"
        return _truncate(out, self.cfg.max_output_chars)

    def glob_files(self, pattern: str, path: str = ".") -> str:
        """Find files by glob pattern (supports ** for recursive matching),
        e.g. "**/*.tsx" or "src/**/test_*.py". Results are sorted newest
        first, like Claude Code's Glob tool, skipping noise directories."""
        root = _resolve_in_scope(self.cfg.project_root, path)
        matches = globmod.glob(os.path.join(root, pattern), recursive=True)

        files = []
        for m in matches:
            if not os.path.isfile(m):
                continue
            rel = os.path.relpath(m, self.cfg.project_root)
            if any(part in _IGNORE_DIRS for part in rel.split(os.sep)):
                continue
            files.append((m, rel))

        if not files:
            return "(no matches)"

        files.sort(key=lambda pair: os.path.getmtime(pair[0]), reverse=True)
        rels = [rel for _, rel in files]
        limit = 200
        out = "\n".join(rels[:limit])
        if len(rels) > limit:
            out += f"\n...[{len(rels) - limit} more matches truncated — narrow the pattern]"
        return out

    def git_diff(self, path: str = ".") -> str:
        p = _resolve_in_scope(self.cfg.project_root, path)
        try:
            result = subprocess.run(
                ["git", "diff"], cwd=p, capture_output=True, text=True, timeout=15,
            )
            return _truncate(result.stdout or "(no changes)", self.cfg.max_output_chars)
        except Exception as e:
            return f"ERROR: {e}"

    def git_status(self, path: str = ".") -> str:
        p = _resolve_in_scope(self.cfg.project_root, path)
        try:
            result = subprocess.run(
                ["git", "status", "--short", "--branch"], cwd=p, capture_output=True, text=True, timeout=15,
            )
            return _truncate(result.stdout or "(clean)", self.cfg.max_output_chars)
        except Exception as e:
            return f"ERROR: {e}"

    def git_log(self, path: str = ".", max_count: int = 20) -> str:
        p = _resolve_in_scope(self.cfg.project_root, path)
        try:
            result = subprocess.run(
                ["git", "log", f"-{max_count}", "--pretty=format:%h %ad %an: %s", "--date=short"],
                cwd=p, capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                return f"ERROR: {result.stderr.strip()}"
            return _truncate(result.stdout or "(no commits)", self.cfg.max_output_chars)
        except Exception as e:
            return f"ERROR: {e}"

    def git_show(self, ref: str = "HEAD", path: str = ".") -> str:
        p = _resolve_in_scope(self.cfg.project_root, path)
        try:
            result = subprocess.run(
                ["git", "show", ref], cwd=p, capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                return f"ERROR: {result.stderr.strip()}"
            return _truncate(result.stdout, self.cfg.max_output_chars)
        except Exception as e:
            return f"ERROR: {e}"

    def git_branch(self, path: str = ".") -> str:
        p = _resolve_in_scope(self.cfg.project_root, path)
        try:
            result = subprocess.run(
                ["git", "branch", "-vv"], cwd=p, capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                return f"ERROR: {result.stderr.strip()}"
            return _truncate(result.stdout or "(no branches)", self.cfg.max_output_chars)
        except Exception as e:
            return f"ERROR: {e}"

    def git_fetch(self, remote: str = "origin") -> str:
        """Update remote-tracking refs without touching the working tree —
        safe to run without approval, unlike pull/push."""
        try:
            result = subprocess.run(
                ["git", "fetch", remote], cwd=self.cfg.project_root,
                capture_output=True, text=True, timeout=self.cfg.shell_timeout_s,
            )
            out = f"exit_code: {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            return _truncate(out, self.cfg.max_output_chars)
        except subprocess.TimeoutExpired:
            return f"ERROR: git fetch timed out after {self.cfg.shell_timeout_s}s"
        except Exception as e:
            return f"ERROR: {e}"

    # ---- write tools (require approval unless auto_approve) ----

    def git_add(self, paths: str = ".") -> str:
        """Stage files for commit. `paths` is a space-separated list of file
        paths relative to the project root, or "." to stage all changes."""
        if paths.strip() == ".":
            args = ["git", "add", "-A"]
        else:
            resolved = []
            for part in paths.split():
                p = _resolve_in_scope(self.cfg.project_root, part)
                resolved.append(os.path.relpath(p, self.cfg.project_root))
            args = ["git", "add"] + resolved
        try:
            result = subprocess.run(
                args, cwd=self.cfg.project_root, capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                return f"ERROR: {result.stderr.strip()}"
            return f"Staged: {paths}"
        except Exception as e:
            return f"ERROR: {e}"

    def git_commit(self, message: str) -> str:
        if not message.strip():
            return "ERROR: commit message cannot be empty"
        try:
            result = subprocess.run(
                ["git", "commit", "-m", message], cwd=self.cfg.project_root,
                capture_output=True, text=True, timeout=15,
            )
            out = f"exit_code: {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            return _truncate(out, self.cfg.max_output_chars)
        except Exception as e:
            return f"ERROR: {e}"

    def git_pull(self, remote: str = "origin", branch: str = "") -> str:
        """Fetch and merge from a remote into the current branch. Touches
        the working tree, so this always requires approval (never auto_approve
        -exempt like the read-only git tools)."""
        args = ["git", "pull", remote]
        if branch:
            args.append(branch)
        try:
            result = subprocess.run(
                args, cwd=self.cfg.project_root, capture_output=True, text=True,
                timeout=self.cfg.shell_timeout_s,
            )
            out = f"exit_code: {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            return _truncate(out, self.cfg.max_output_chars)
        except subprocess.TimeoutExpired:
            return f"ERROR: git pull timed out after {self.cfg.shell_timeout_s}s"
        except Exception as e:
            return f"ERROR: {e}"

    def git_push(self, remote: str = "origin", branch: str = "") -> str:
        """Push commits to a remote. No force-push option is exposed —
        this tool only ever does a plain push."""
        args = ["git", "push", remote]
        if branch:
            args.append(branch)
        try:
            result = subprocess.run(
                args, cwd=self.cfg.project_root, capture_output=True, text=True,
                timeout=self.cfg.shell_timeout_s,
            )
            out = f"exit_code: {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            return _truncate(out, self.cfg.max_output_chars)
        except subprocess.TimeoutExpired:
            return f"ERROR: git push timed out after {self.cfg.shell_timeout_s}s"
        except Exception as e:
            return f"ERROR: {e}"

    def write_file(self, path: str, content: str, overwrite: bool = False) -> str:
        p = _resolve_in_scope(self.cfg.project_root, path)
        if os.path.exists(p) and not overwrite:
            return (f"ERROR: {path} already exists. Use edit_file to modify it, "
                     f"or pass overwrite=true to replace it entirely.")
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} chars to {path}"

    def preview_edit(self, path: str, old_str: str, new_str: str):
        """Dry-run version of edit_file — validates and computes the diff
        WITHOUT writing anything. Returns (ok: bool, message: str) where
        message is either the diff (ok=True) or an error (ok=False)."""
        p = _resolve_in_scope(self.cfg.project_root, path)
        if not os.path.isfile(p):
            return False, f"{path} does not exist"
        with open(p, "r", encoding="utf-8") as f:
            original = f.read()
        count = original.count(old_str)
        if count == 0:
            return False, f"old_str not found in {path}"
        if count > 1:
            return False, f"old_str matches {count} locations in {path} — not unique"
        updated = original.replace(old_str, new_str, 1)
        diff = "\n".join(difflib.unified_diff(
            original.splitlines(), updated.splitlines(),
            fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="",
        ))
        return True, diff

    def preview_write(self, path: str, content: str, overwrite: bool = False):
        """Dry-run version of write_file. Returns (is_new: bool, preview: str).
        If the file exists and overwrite=True, preview is a diff; if it's a
        new file, preview is the content itself (syntax-highlighted by the UI)."""
        p = _resolve_in_scope(self.cfg.project_root, path)
        if os.path.isfile(p) and overwrite:
            with open(p, "r", encoding="utf-8") as f:
                original = f.read()
            diff = "\n".join(difflib.unified_diff(
                original.splitlines(), content.splitlines(),
                fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="",
            ))
            return False, diff
        return True, content

    def edit_file(self, path: str, old_str: str, new_str: str) -> str:
        """Find-and-replace that requires an exact, unique match — same
        discipline as Claude's own str_replace tool, to avoid silently
        editing the wrong spot."""
        p = _resolve_in_scope(self.cfg.project_root, path)
        if not os.path.isfile(p):
            return f"ERROR: {path} does not exist"
        with open(p, "r", encoding="utf-8") as f:
            original = f.read()

        count = original.count(old_str)
        if count == 0:
            return f"ERROR: old_str not found in {path}. Nothing changed."
        if count > 1:
            return (f"ERROR: old_str matches {count} locations in {path}. "
                     f"Include more surrounding context to make it unique.")

        updated = original.replace(old_str, new_str, 1)
        with open(p, "w", encoding="utf-8") as f:
            f.write(updated)

        diff = "\n".join(difflib.unified_diff(
            original.splitlines(), updated.splitlines(),
            fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="",
        ))
        return f"Edited {path}.\n{_truncate(diff, self.cfg.max_output_chars)}"

    def run_shell(self, command: str) -> str:
        for bad in self.cfg.denied_shell_patterns:
            if bad in command:
                return f"ERROR: command blocked by policy (matched pattern: '{bad}')"
        try:
            result = subprocess.run(
                command, shell=True, cwd=self.cfg.project_root,
                capture_output=True, text=True, timeout=self.cfg.shell_timeout_s,
            )
            out = f"exit_code: {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            return _truncate(out, self.cfg.max_output_chars)
        except subprocess.TimeoutExpired:
            return f"ERROR: command timed out after {self.cfg.shell_timeout_s}s"
        except Exception as e:
            return f"ERROR: {e}"

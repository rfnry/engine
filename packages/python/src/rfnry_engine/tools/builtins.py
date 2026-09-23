from __future__ import annotations

import asyncio
import time
from itertools import islice
from pathlib import Path
from typing import Any, Literal

import regex
from pydantic import BaseModel, Field

from rfnry_engine.errors import InputError
from rfnry_engine.frontmatter import strip_frontmatter
from rfnry_engine.jail import Jail
from rfnry_engine.tools.tools import Tool

_MAX_BYTES = 5 * 1024 * 1024
_MAX_WALK = 100_000
_HEAD_LIMIT = 250
_GREP_BYTES = 50 * 1024 * 1024
_GREP_SECONDS = 10.0
_SNIFF = 8192


class ReadArgs(BaseModel):
    path: str
    offset: int = Field(default=0, ge=0)
    limit: int | None = Field(default=None, ge=1)


class GrepArgs(BaseModel):
    pattern: str
    path: str | None = None
    glob: str | None = None
    case_insensitive: bool = False
    mode: Literal["files", "content"] = "files"
    head_limit: int = Field(default=_HEAD_LIMIT, ge=1)


class GlobArgs(BaseModel):
    pattern: str


class LSArgs(BaseModel):
    path: str | None = None
    limit: int = Field(default=_HEAD_LIMIT, ge=1)


class SkillArgs(BaseModel):
    name: str


def _hidden(rel: Path) -> bool:
    return any(part.startswith(".") for part in rel.parts)


def _text(path: Path) -> str | None:
    data = path.read_bytes()
    if b"\0" in data[:_SNIFF]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


class _Directory:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.jail = Jail(self.root)

    def _inside(self, path: Path) -> bool:
        try:
            self.jail.resolve(path)
        except InputError:
            return False
        return True

    def read(self, args: dict[str, Any]) -> str:
        a = ReadArgs.model_validate(args)
        target = self.jail.resolve(a.path)
        if not target.is_file():
            raise InputError(f"builtins: not a file: {a.path}")
        if target.stat().st_size > _MAX_BYTES:
            raise InputError(f"builtins: file exceeds {_MAX_BYTES} bytes: {a.path}")
        text = _text(target)
        if text is None:
            raise InputError(f"builtins: binary file: {a.path}; Read handles text only, reading it needs a declared tool")
        if a.offset or a.limit is not None:
            lines = text.splitlines()
            end = a.offset + a.limit if a.limit is not None else len(lines)
            text = "\n".join(lines[a.offset : end])
        return text

    def _files_under(self, base: Path, glob: str | None) -> list[Path]:
        if base.is_file():
            return [base]
        found: list[Path] = []
        for candidate in islice(base.rglob("*"), _MAX_WALK):
            rel = candidate.relative_to(self.root)
            if not candidate.is_file() or _hidden(rel) or not self._inside(candidate):
                continue
            if glob is not None and not rel.match(glob):
                continue
            found.append(candidate)
        return sorted(found)

    def grep(self, args: dict[str, Any]) -> str:
        a = GrepArgs.model_validate(args)
        try:
            pattern = regex.compile(a.pattern, regex.IGNORECASE if a.case_insensitive else 0)
        except regex.error as exc:
            raise InputError(f"builtins: invalid pattern: {exc}") from exc
        budget = _GREP_SECONDS
        deadline = time.monotonic() + budget

        def search(text: str) -> bool:
            try:
                return pattern.search(text, timeout=max(deadline - time.monotonic(), 0.0)) is not None
            except TimeoutError as exc:
                raise InputError(f"builtins: pattern took longer than {budget:g}s; simplify it or narrow path and glob") from exc

        base = self.jail.resolve(a.path) if a.path else self.root
        lines: list[str] = []
        scanned = 0
        exhausted = False
        for path in self._files_under(base, a.glob):
            rel = path.relative_to(self.root)
            try:
                size = path.stat().st_size
                if size > _MAX_BYTES:
                    continue
                if scanned + size > _GREP_BYTES:
                    exhausted = True
                    break
                scanned += size
                text = _text(path)
            except OSError:
                continue
            if text is None:
                continue
            if a.mode == "files":
                if search(text):
                    lines.append(str(rel))
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if search(line):
                    lines.append(f"{rel}:{number}:{line}")
        notes: list[str] = []
        if len(lines) > a.head_limit:
            notes.append(f"[truncated to {a.head_limit} of {len(lines)}]")
        if exhausted:
            notes.append(f"[truncated: scan budget of {_GREP_BYTES} bytes reached]")
        return "\n".join(lines[: a.head_limit] or ["no matches found"]) + "".join(f"\n{note}" for note in notes)

    def glob(self, args: dict[str, Any]) -> str:
        a = GlobArgs.model_validate(args)
        if a.pattern.startswith("/") or ".." in a.pattern.split("/"):
            raise InputError("builtins: pattern must be relative and stay inside the agent directory")
        entries = list(islice(self.root.glob(a.pattern), _MAX_WALK + 1))
        truncated = len(entries) > _MAX_WALK
        matches = sorted(
            str(p.relative_to(self.root))
            for p in entries[:_MAX_WALK]
            if p.is_file() and not _hidden(p.relative_to(self.root)) and self._inside(p)
        )
        suffix = f"\n[truncated: walk exceeded {_MAX_WALK} entries]" if truncated else ""
        return "\n".join(matches) + suffix

    def ls(self, args: dict[str, Any]) -> str:
        a = LSArgs.model_validate(args)
        target = self.jail.resolve(a.path) if a.path else self.root
        if not target.is_dir():
            raise InputError(f"builtins: not a directory: {a.path}")
        entries = [p for p in sorted(target.iterdir(), key=lambda p: p.name) if not p.name.startswith(".")]
        if not entries:
            return "(empty)"
        shown = [f"{p.name}/" if p.is_dir() else p.name for p in entries[: a.limit]]
        suffix = f"\n[truncated to {a.limit} of {len(entries)}]" if len(entries) > a.limit else ""
        return "\n".join(shown) + suffix

    def skill(self, args: dict[str, Any]) -> str:
        a = SkillArgs.model_validate(args)
        flat = self.root / "skills" / f"{a.name}.md"
        folder = self.root / "skills" / a.name / "SKILL.md"
        target = Jail(self.root / "skills").resolve(flat if flat.is_file() else folder)
        if not target.is_file():
            raise InputError(f"builtins: skill not found: {a.name}")
        return strip_frontmatter(target.read_text(encoding="utf-8")).lstrip("\n").rstrip()


def builtins(root: Path) -> list[Tool]:
    directory = _Directory(root)

    def threaded(handler: Any) -> Any:
        async def body(args: dict[str, Any]) -> str:
            result: str = await asyncio.to_thread(handler, args)
            return result

        return body

    return [
        Tool("Read", "Read a file in the agent directory. Optional offset/limit select lines.", body=threaded(directory.read), args=ReadArgs),
        Tool("Grep", "Search file contents with a regex. mode=files lists matching files; mode=content lists matching lines.", body=threaded(directory.grep), args=GrepArgs),
        Tool("Glob", "Find files by glob pattern, e.g. '**/*.md'. Paths are relative to the agent directory.", body=threaded(directory.glob), args=GlobArgs),
        Tool("LS", "List a directory inside the agent directory.", body=threaded(directory.ls), args=LSArgs),
        Tool("Skill", "Load a skill's body by name from skills/.", body=threaded(directory.skill), args=SkillArgs),
    ]

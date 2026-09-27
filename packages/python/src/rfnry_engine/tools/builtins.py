from __future__ import annotations

import asyncio
import mimetypes
import time
from collections.abc import Collection
from fnmatch import fnmatch
from itertools import islice
from pathlib import Path
from typing import Any, Literal

import regex
from pydantic import BaseModel, Field

from rfnry_engine.errors import InputError
from rfnry_engine.frontmatter import strip_frontmatter
from rfnry_engine.jail import Jail
from rfnry_engine.models import Audio, Document, Image, Part
from rfnry_engine.tools.tools import Result, Tool

_MAX_BYTES = 5 * 1024 * 1024
_LOAD_BYTES = 20 * 1024 * 1024
_MAX_WALK = 100_000
_HEAD_LIMIT = 250
_GREP_BYTES = 50 * 1024 * 1024
_GREP_SECONDS = 10.0
_SNIFF = 8192


class ReadArgs(BaseModel):
    path: str
    offset: int = Field(default=0, ge=0)
    limit: int | None = Field(default=None, ge=1)


class LoadArgs(BaseModel):
    path: str


class LoadAnywhereArgs(BaseModel):
    path: str
    where: Literal["folder", "workspace"] = "folder"


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
    return _decoded(path.read_bytes())


_SIGNATURES = {
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/gif": (b"GIF87a", b"GIF89a"),
    "application/pdf": (b"%PDF-",),
}


def _signed(data: bytes) -> str | None:
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return next((kind for kind, marks in _SIGNATURES.items() if data.startswith(marks)), None)


def _media_type(path: Path, data: bytes) -> str:
    named = mimetypes.guess_type(path.name)[0]
    signed = _signed(data)
    if named is None:
        return signed or "application/octet-stream"
    if (named in _SIGNATURES or named == "image/webp") and signed != named:
        raise InputError(f"builtins: {path.name} is not the {named} its name says")
    return named


def _part(path: Path, data: bytes, media_type: str) -> Part:
    if media_type.startswith("image/"):
        return Image(media_type=media_type, data=data)
    if media_type.startswith("audio/"):
        return Audio(media_type=media_type, data=data)
    return Document(media_type=media_type, name=path.name, data=data)


def _decoded(data: bytes) -> str | None:
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

    def read(self, args: dict[str, Any]) -> Result:
        a = ReadArgs.model_validate(args)
        target = self.jail.resolve(a.path)
        if not target.is_file():
            raise InputError(f"builtins: not a file: {a.path}")
        if target.stat().st_size > _MAX_BYTES:
            raise InputError(f"builtins: file exceeds {_MAX_BYTES} bytes: {a.path}")
        text = _decoded(target.read_bytes())
        if text is None:
            raise InputError(
                f"builtins: {a.path} is not a text file; Load gives you the file itself"
            )
        if a.offset or a.limit is not None:
            lines = text.splitlines()
            end = a.offset + a.limit if a.limit is not None else len(lines)
            text = "\n".join(lines[a.offset : end])
        return text

    def _files_under(self, base: Path, glob: str | None) -> tuple[list[Path], bool]:
        if base.is_file():
            return [base], False
        walked = list(islice(base.rglob("*"), _MAX_WALK + 1))
        found: list[Path] = []
        for candidate in walked[:_MAX_WALK]:
            rel = candidate.relative_to(self.root)
            if not candidate.is_file() or _hidden(rel) or not self._inside(candidate):
                continue
            if glob is not None and not rel.match(glob):
                continue
            found.append(candidate)
        return sorted(found), len(walked) > _MAX_WALK

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
                return (
                    pattern.search(text, timeout=max(deadline - time.monotonic(), 0.0)) is not None
                )
            except TimeoutError as exc:
                raise InputError(
                    f"builtins: pattern took longer than {budget:g}s; simplify it or narrow path and glob"
                ) from exc

        base = self.jail.resolve(a.path) if a.path else self.root
        lines: list[str] = []
        scanned = 0
        exhausted = False
        paths, walked_out = self._files_under(base, a.glob)
        for path in paths:
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
        if walked_out:
            notes.append(f"[truncated: walk exceeded {_MAX_WALK} entries]")
        return "\n".join(lines[: a.head_limit] or ["no matches found"]) + "".join(
            f"\n{note}" for note in notes
        )

    def glob(self, args: dict[str, Any]) -> str:
        a = GlobArgs.model_validate(args)
        if a.pattern.startswith("/") or ".." in a.pattern.split("/"):
            raise InputError(
                "builtins: pattern must be relative and stay inside the agent directory"
            )
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
        entries = [
            p for p in sorted(target.iterdir(), key=lambda p: p.name) if not p.name.startswith(".")
        ]
        if not entries:
            return "(empty)"
        shown = [f"{p.name}/" if p.is_dir() else p.name for p in entries[: a.limit]]
        suffix = f"\n[truncated to {a.limit} of {len(entries)}]" if len(entries) > a.limit else ""
        return "\n".join(shown) + suffix

    def skill(self, args: dict[str, Any]) -> str:
        a = SkillArgs.model_validate(args)
        flat = self.root / "skills" / f"{a.name}.md"
        folder = self.root / "skills" / a.name / "SKILL.md"
        try:
            target = Jail(self.root / "skills").resolve(flat if flat.is_file() else folder)
        except InputError:
            raise InputError(f"builtins: skill outside skills/: {a.name}") from None
        if not target.is_file():
            raise InputError(f"builtins: skill not found: {a.name}")
        return strip_frontmatter(target.read_text(encoding="utf-8")).lstrip("\n").rstrip()


def builtins(root: Path) -> list[Tool]:
    directory = _Directory(root)

    def threaded(handler: Any) -> Any:
        async def body(args: dict[str, Any]) -> Result:
            result: Result = await asyncio.to_thread(handler, args)
            return result

        return body

    return [
        Tool(
            "Read",
            "Read a text file in the agent directory. Optional offset/limit select lines. For an image, a PDF, audio or any other file, use Load.",
            body=threaded(directory.read),
            args=ReadArgs,
        ),
        Tool(
            "Grep",
            "Search file contents with a regex. mode=files lists matching files; mode=content lists matching lines.",
            body=threaded(directory.grep),
            args=GrepArgs,
        ),
        Tool(
            "Glob",
            "Find files by glob pattern, e.g. '**/*.md'. Paths are relative to the agent directory.",
            body=threaded(directory.glob),
            args=GlobArgs,
        ),
        Tool(
            "LS",
            "List a directory inside the agent directory.",
            body=threaded(directory.ls),
            args=LSArgs,
        ),
        Tool(
            "Skill",
            "Load a skill's body by name from skills/.",
            body=threaded(directory.skill),
            args=SkillArgs,
        ),
    ]


def load_tool(
    folder: Path, workspace: Path | None = None, media_types: Collection[str] | None = None
) -> Tool:
    places = {"folder": Jail(folder)}
    if workspace is not None:
        places["workspace"] = Jail(workspace)

    def load(args: dict[str, Any]) -> list[Part]:
        a = LoadAnywhereArgs.model_validate(args)
        try:
            target = places[a.where].resolve(a.path)
        except InputError:
            if a.where == "folder":
                raise
            raise InputError(f"builtins: path outside the workspace: {a.path}") from None
        if not target.is_file():
            raise InputError(f"builtins: not a file: {a.path}")
        if target.stat().st_size > _LOAD_BYTES:
            raise InputError(f"builtins: file exceeds {_LOAD_BYTES} bytes: {a.path}")
        data = target.read_bytes()
        media_type = _media_type(target, data)
        if media_types is not None and not any(fnmatch(media_type, t) for t in media_types):
            raise InputError(f"builtins: the model does not take {media_type} files: {a.path}")
        return [_part(target, data, media_type)]

    async def body(args: dict[str, Any]) -> Result:
        parts: list[Part] = await asyncio.to_thread(load, args)
        return parts

    if workspace is None:
        return Tool(
            "Load",
            "Load a file from the agent directory as it is, such as an image, a PDF or audio, so you see or hear it yourself.",
            body=body,
            args=LoadArgs,
        )
    return Tool(
        "Load",
        "Load a file as it is, such as an image, a PDF or audio, so you see or hear it yourself. "
        "where is 'folder' for the agent directory or 'workspace' for your workspace.",
        body=body,
        args=LoadAnywhereArgs,
    )

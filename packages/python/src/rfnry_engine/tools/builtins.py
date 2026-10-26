import asyncio
import errno
import mimetypes
import os
import stat
import time
from collections.abc import Collection, Iterable, Mapping
from fnmatch import fnmatch
from itertools import islice
from pathlib import Path
from typing import Any, Literal

import regex
from pydantic import BaseModel, Field

from rfnry_engine.errors import InputError
from rfnry_engine.jail import Jail
from rfnry_engine.models import Audio, Document, Image, Part, Text
from rfnry_engine.tools.tools import Result, Tool
from rfnry_engine.tools.workspace import Backend

BUILTINS = frozenset({"Read", "Grep", "Glob", "LS", "Load"})

_MAX_BYTES = 5 * 1024 * 1024
_LOAD_BYTES = 20 * 1024 * 1024
_MAX_WALK = 100_000
_HEAD_LIMIT = 250
_GREP_BYTES = 50 * 1024 * 1024
_GREP_SECONDS = 10.0
_SNIFF = 8192
_OPEN_FILES = "/proc/self/fd"


class ReadArgs(BaseModel):
    path: str
    offset: int = Field(default=0, ge=0)
    limit: int | None = Field(default=None, ge=1)


class ReadAnywhereArgs(ReadArgs):
    where: Literal["folder", "workspace"] = "folder"


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


def _hidden(rel: Path) -> bool:
    return any(
        (part.startswith(".") and part != "..") or part == "__pycache__" for part in rel.parts
    )


def _within(jail: Jail, path: Path) -> Path | None:
    real = Path(os.path.realpath(path))
    root = next((r for r in (jail.root, *jail.also) if real == r or r in real.parents), None)
    return None if root is None else real.relative_to(root)


def _named(jail: Jail, path: str) -> str:
    named = jail.root / path
    seen = _within(jail, named)
    if seen is None:
        return path
    folder = _within(jail, named.parent)
    if folder is not None and not os.path.isdir(named):
        seen = folder / named.name
    if _hidden(seen):
        raise InputError(
            f"tools/builtins: {path} is hidden; names starting with . and __pycache__ are not part of the agent's folder"
        )
    return path


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
        raise InputError(f"tools/builtins: {path.name} is not the {named} its name says")
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


def _handed(path: Path, data: bytes, shown: str, media_types: Collection[str] | None) -> Part:
    media_type = _media_type(path, data)
    if media_types is not None and not any(fnmatch(media_type, t) for t in media_types):
        raise InputError(f"tools/builtins: the model does not take {media_type} files: {shown}")
    return _part(path, data, media_type)


def preload_parts(
    names: list[str], media_types: Collection[str] | None, jail: Jail
) -> tuple[list[Part], dict[str, bytes]]:
    parts: list[Part] = []
    read: dict[str, bytes] = {}
    for name in names:
        opened, data = _contents(jail, name, _LOAD_BYTES)
        read[name] = data
        text = _decoded(data) if len(data) <= _MAX_BYTES else None
        if text is None:
            parts += [Text(text=f"<<<{name}>>>"), _handed(opened, data, name, media_types)]
        else:
            parts.append(Text(text=f"<<<{name}>>>\n{text}"))
    return parts, read


def _name_of(fd: int, path: str, removed: bool) -> Path:
    try:
        opened = os.readlink(f"{_OPEN_FILES}/{fd}")
    except OSError:
        return Path(path)
    return Path(opened.removesuffix(" (deleted)") if removed else opened)


def _contents(jail: Jail, path: str, limit: int, missing: str = "") -> tuple[Path, bytes]:
    try:
        fd = jail.open(path)
        try:
            found = os.fstat(fd)
            if not stat.S_ISREG(found.st_mode):
                raise InputError(f"tools/builtins: not a file: {path}")
            if found.st_size > limit:
                raise InputError(f"tools/builtins: file exceeds {limit} bytes: {path}")
            opened = _name_of(fd, path, found.st_nlink == 0)
            with open(fd, "rb", closefd=False) as file:
                data = file.read(limit + 1)
        finally:
            os.close(fd)
    except OSError as exc:
        if exc.errno == errno.ENOENT and missing:
            raise InputError(
                f"tools/builtins: not a file of your folder: {path}; {missing}"
            ) from None
        if exc.errno in (errno.ENOENT, errno.ENOTDIR, errno.ENXIO):
            raise InputError(f"tools/builtins: not a file: {path}") from None
        raise InputError(f"tools/builtins: cannot read {path}: {exc.strerror}") from None
    if len(data) > limit:
        raise InputError(f"tools/builtins: file exceeds {limit} bytes: {path}")
    return opened, data


def _lines(data: bytes, a: ReadArgs) -> str:
    text = _decoded(data)
    if text is None:
        raise InputError(
            f"tools/builtins: {a.path} is not a text file; Load gives you the file itself"
        )
    if a.offset or a.limit is not None:
        lines = text.splitlines()
        end = a.offset + a.limit if a.limit is not None else len(lines)
        text = "\n".join(lines[a.offset : end])
    return text


def _read_tool(read: Tool, workspace: Backend | None) -> Tool:
    if workspace is None:
        return read

    async def body(args: dict[str, Any]) -> Result:
        a = ReadAnywhereArgs.model_validate(args)
        if a.where == "folder":
            return await read.call(args)
        return _lines(await workspace.read(a.path, _MAX_BYTES), a)

    return Tool(
        "Read",
        f"{read.description} where is 'folder' for a file of the FILES map or 'workspace' for your workspace.",
        body=body,
        args=ReadAnywhereArgs,
    )


class _Directory:
    def __init__(
        self, root: Path, reads: Mapping[str, Path] | None = None, workspace: bool = False
    ) -> None:
        self.root = Path(root).resolve()
        self.workspace = workspace
        self.reads = {name: Path(folder).resolve() for name, folder in (reads or {}).items()}
        self.jail = Jail(self.root, also=self.reads.values())

    def _shown(self, path: Path) -> str:
        for name, folder in self.reads.items():
            if folder in path.parents:
                return f"{name}/{path.relative_to(folder).as_posix()}"
        return path.relative_to(self.root).as_posix()

    def _inside(self, path: Path) -> bool:
        try:
            self.jail.resolve(path)
        except InputError:
            return False
        return True

    def read(self, args: dict[str, Any]) -> Result:
        a = ReadArgs.model_validate(args)
        missing = 'Read with where="workspace" reads from your workspace' if self.workspace else ""
        return _lines(_contents(self.jail, _named(self.jail, a.path), _MAX_BYTES, missing)[1], a)

    def _files_under(self, base: Path, glob: str | None) -> tuple[list[Path], bool]:
        if base.is_file():
            return [base], False
        walked = list(islice(base.rglob("*"), _MAX_WALK + 1))
        found: list[Path] = []
        for candidate in walked[:_MAX_WALK]:
            rel = Path(self._shown(candidate))
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
            raise InputError(f"tools/builtins: invalid pattern: {exc}") from exc
        budget = _GREP_SECONDS
        deadline = time.monotonic() + budget

        def search(text: str) -> bool:
            try:
                return (
                    pattern.search(text, timeout=max(deadline - time.monotonic(), 0.0)) is not None
                )
            except TimeoutError as exc:
                raise InputError(
                    f"tools/builtins: pattern took longer than {budget:g}s; simplify it or narrow path and glob"
                ) from exc

        base = self.jail.resolve(_named(self.jail, a.path)) if a.path else self.root
        lines: list[str] = []
        scanned = 0
        exhausted = False
        try:
            paths, walked_out = self._files_under(base, a.glob)
        except ValueError as exc:
            raise InputError(f"tools/builtins: invalid glob {a.glob!r}: {exc}") from exc
        for path in paths:
            rel = self._shown(path)
            try:
                fd = self.jail.open(path)
            except InputError, OSError:
                continue
            try:
                found = os.fstat(fd)
                if not stat.S_ISREG(found.st_mode) or found.st_size > _MAX_BYTES:
                    continue
                if scanned + found.st_size > _GREP_BYTES:
                    exhausted = True
                    break
                scanned += found.st_size
                with open(fd, "rb", closefd=False) as file:
                    data = file.read(_MAX_BYTES + 1)
            except OSError:
                continue
            finally:
                os.close(fd)
            text = _decoded(data) if len(data) <= _MAX_BYTES else None
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
        base, pattern = self.root, a.pattern
        for name, folder in self.reads.items():
            if pattern.startswith(f"{name}/"):
                base, pattern = folder, pattern[len(name) + 1 :]
                break
        if not pattern or pattern.startswith("/") or ".." in pattern.split("/"):
            raise InputError(
                "tools/builtins: pattern must be relative and stay inside the agent directory or a folder it reads"
            )
        try:
            entries = list(islice(base.glob(pattern), _MAX_WALK + 1))
        except ValueError as exc:
            raise InputError(f"tools/builtins: invalid glob pattern {a.pattern!r}: {exc}") from exc
        truncated = len(entries) > _MAX_WALK
        matches = sorted(
            self._shown(p)
            for p in entries[:_MAX_WALK]
            if p.is_file() and not _hidden(Path(self._shown(p))) and self._inside(p)
        )
        suffix = f"\n[truncated: walk exceeded {_MAX_WALK} entries]" if truncated else ""
        return "\n".join(matches) + suffix

    def ls(self, args: dict[str, Any]) -> str:
        a = LSArgs.model_validate(args)
        path = _named(self.jail, a.path or ".")
        try:
            fd = self.jail.open(path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                with os.scandir(fd) as found:
                    entries = sorted(
                        (e for e in found if not _hidden(Path(e.name))), key=lambda e: e.name
                    )
                shown = [f"{e.name}/" if e.is_dir() else e.name for e in entries[: a.limit]]
            finally:
                os.close(fd)
        except OSError as exc:
            if exc.errno == errno.ENOENT and self.workspace:
                raise InputError(
                    f"tools/builtins: not a directory of your folder: {path}; Shell lists your workspace"
                ) from None
            if exc.errno in (errno.ENOENT, errno.ENOTDIR):
                raise InputError(f"tools/builtins: not a directory: {path}") from None
            raise InputError(f"tools/builtins: cannot list {path}: {exc.strerror}") from None
        if not entries:
            return "(empty)"
        suffix = f"\n[truncated to {a.limit} of {len(entries)}]" if len(entries) > a.limit else ""
        return "\n".join(shown) + suffix


def builtins(
    root: Path, reads: Mapping[str, Path] | None = None, workspace: Backend | None = None
) -> list[Tool]:
    directory = _Directory(root, reads, workspace is not None)

    def threaded(handler: Any) -> Any:
        async def body(args: dict[str, Any]) -> Result:
            result: Result = await asyncio.to_thread(handler, args)
            return result

        return body

    return [
        _read_tool(
            Tool(
                "Read",
                "Read a text file by its path in the FILES map. Optional offset/limit select lines. For an image, a PDF, audio or any other file, use Load.",
                body=threaded(directory.read),
                args=ReadArgs,
            ),
            workspace,
        ),
        Tool(
            "Grep",
            "Search file contents with a regex. mode=files lists matching files; mode=content lists matching lines. path is a file or folder of the FILES map; without it, your own folder is searched.",
            body=threaded(directory.grep),
            args=GrepArgs,
        ),
        Tool(
            "Glob",
            "Find files by glob pattern, e.g. '**/*.md'. Paths are those of the FILES map.",
            body=threaded(directory.glob),
            args=GlobArgs,
        ),
        Tool(
            "LS",
            "List a folder by its path in the FILES map; without a path, your own folder.",
            body=threaded(directory.ls),
            args=LSArgs,
        ),
    ]


def load_tool(
    folder: Path,
    workspace: Backend | None = None,
    media_types: Collection[str] | None = None,
    reads: Iterable[Path] = (),
) -> Tool:
    jail = Jail(folder, also=reads)
    missing = (
        'Load with where="workspace" loads from your workspace' if workspace is not None else ""
    )

    def load(path: str) -> list[Part]:
        opened, data = _contents(jail, _named(jail, path), _LOAD_BYTES, missing)
        return [_handed(opened, data, path, media_types)]

    async def body(args: dict[str, Any]) -> Result:
        a = LoadAnywhereArgs.model_validate(args)
        if a.where == "workspace" and workspace is not None:
            data = await workspace.read(a.path, _LOAD_BYTES)
            return [_handed(Path(a.path), data, a.path, media_types)]
        parts: list[Part] = await asyncio.to_thread(load, a.path)
        return parts

    if workspace is None:
        return Tool(
            "Load",
            "Load a file of the FILES map as it is, such as an image, a PDF or audio, so you see or hear it yourself.",
            body=body,
            args=LoadArgs,
        )
    return Tool(
        "Load",
        "Load a file as it is, such as an image, a PDF or audio, so you see or hear it yourself. "
        "where is 'folder' for a file of the FILES map or 'workspace' for your workspace.",
        body=body,
        args=LoadAnywhereArgs,
    )

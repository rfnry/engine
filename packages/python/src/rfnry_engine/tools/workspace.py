from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.jail import Jail
from rfnry_engine.tools.tools import Tool

STREAM_CAP = 30_000


class ShellArgs(BaseModel):
    command: str


class WriteArgs(BaseModel):
    path: str
    content: str


class EditArgs(BaseModel):
    path: str
    old: str
    new: str


class _Capture(asyncio.SubprocessProtocol):
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.heads = {1: bytearray(), 2: bytearray()}
        self.tails = {1: bytearray(), 2: bytearray()}
        self.totals = {1: 0, 2: 0}
        self.open = {1, 2}
        self.exited: asyncio.Future[None] = loop.create_future()
        self.closed: asyncio.Future[None] = loop.create_future()

    def pipe_data_received(self, fd: int, data: bytes) -> None:
        half = STREAM_CAP // 2
        head, tail = self.heads[fd], self.tails[fd]
        self.totals[fd] += len(data)
        room = half - len(head)
        if room > 0:
            head += data[:room]
            data = data[room:]
        tail += data
        if len(tail) > half:
            del tail[: len(tail) - half]

    def pipe_connection_lost(self, fd: int, exc: Exception | None) -> None:
        self.open.discard(fd)
        if not self.open and not self.closed.done():
            self.closed.set_result(None)

    def process_exited(self) -> None:
        if not self.exited.done():
            self.exited.set_result(None)

    def text(self, fd: int) -> str:
        head, tail = bytes(self.heads[fd]), bytes(self.tails[fd])
        omitted = self.totals[fd] - len(head) - len(tail)
        if omitted <= 0:
            return (head + tail).decode("utf-8", errors="replace")
        return (
            head.decode("utf-8", errors="replace")
            + f"\n[... {omitted} bytes omitted ...]\n"
            + tail.decode("utf-8", errors="replace")
        )


def _kill(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)


def _threaded(handler: Callable[[dict[str, Any]], str]) -> Callable[[dict[str, Any]], Awaitable[str]]:
    async def body(args: dict[str, Any]) -> str:
        return await asyncio.to_thread(handler, args)

    return body


class Workspace:
    def __init__(self, root: Path | str, *, env: dict[str, str] | None = None) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ConfigurationError(f"workspace: root is not a directory: {root}")
        self.env = (
            dict(env)
            if env is not None
            else {"PATH": os.environ.get("PATH", os.defpath), "HOME": str(self.root), "LANG": "C.UTF-8"}
        )
        self.jail = Jail(self.root)

    def tools(self) -> list[Tool]:
        return [
            Tool("Shell", "Run a shell command; it already runs in the workspace directory, so relative paths work and there is no need to search the filesystem for it. Returns exit_code, stdout and stderr.", body=self._shell, args=ShellArgs),
            Tool("Write", "Write a file in the workspace with exactly this content, creating parent directories.", body=_threaded(self._write), args=WriteArgs),
            Tool("Edit", "Replace one exact occurrence of old with new in a workspace file.", body=_threaded(self._edit), args=EditArgs),
        ]

    async def _shell(self, args: dict[str, Any]) -> dict[str, Any]:
        command = ShellArgs.model_validate(args).command
        loop = asyncio.get_running_loop()
        transport, capture = await loop.subprocess_exec(
            lambda: _Capture(loop),
            "/bin/sh", "-c", command, cwd=self.root, env=self.env,
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        pid = transport.get_pid()
        try:
            await capture.exited
            _kill(pid)
            await capture.closed
        except BaseException:
            _kill(pid)
            raise
        finally:
            transport.close()
        return {"exit_code": transport.get_returncode(), "stdout": capture.text(1), "stderr": capture.text(2)}

    def _target(self, path: str) -> Path:
        try:
            return self.jail.resolve(path)
        except InputError:
            raise InputError(f"workspace: path outside the workspace: {path}") from None

    def _write(self, args: dict[str, Any]) -> str:
        a = WriteArgs.model_validate(args)
        target = self._target(a.path)
        if target.is_dir():
            raise InputError(f"workspace: is a directory: {a.path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = a.content.encode("utf-8")
        target.write_bytes(data)
        return f"wrote {len(data)} bytes to {target.relative_to(self.root)}"

    def _edit(self, args: dict[str, Any]) -> str:
        a = EditArgs.model_validate(args)
        if not a.old:
            raise InputError("workspace: old must not be empty")
        target = self._target(a.path)
        if not target.is_file():
            raise InputError(f"workspace: not a file: {a.path}")
        try:
            text = target.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            raise InputError(f"workspace: not a UTF-8 text file: {a.path}") from None
        count = text.count(a.old)
        if count != 1:
            raise InputError(f"workspace: old appears {count} times in {a.path}; it must appear exactly once")
        target.write_bytes(text.replace(a.old, a.new, 1).encode("utf-8"))
        return f"edited {target.relative_to(self.root)}"

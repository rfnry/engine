import asyncio
import hmac
import json
import secrets
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

LINE = 1_048_576

Answer = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]

_BAD = 'tools/bridge: a request is one JSON line: {"key": …, "tool": …, "args": {…}}'


class Bridge:
    def __init__(self, path: Path, key: str, answer: Answer) -> None:
        self.path = path
        self.key = key
        self._answer = answer
        self._handlers: set[asyncio.Task[Any]] = set()
        self._server: asyncio.Server | None = None
        self._closed = False

    @classmethod
    async def open(cls, root: Path, answer: Answer) -> Bridge:
        bridge = cls(root / f".rfnry-{secrets.token_hex(4)}.sock", secrets.token_hex(16), answer)
        bridge._server = await asyncio.start_unix_server(
            bridge._handle, path=bridge.path, limit=LINE
        )
        return bridge

    @property
    def env(self) -> dict[str, str]:
        return {"RFNRY_TOOLS": str(self.path), "RFNRY_TOOLS_KEY": self.key}

    async def _reply(self, line: bytes | None) -> dict[str, Any]:
        if line is None:
            return {"error": f"tools/bridge: a request is at most {LINE} bytes"}
        try:
            request = json.loads(line.decode())
        except ValueError, RecursionError:
            return {"error": _BAD}
        if not isinstance(request, dict):
            return {"error": _BAD}
        key = request.get("key")
        if not isinstance(key, str) or not hmac.compare_digest(
            key.encode("utf-8", "surrogatepass"), self.key.encode()
        ):
            return {"error": "tools/bridge: the key is not this command's"}
        tool, args = request.get("tool"), request.get("args")
        if not isinstance(tool, str) or not isinstance(args, dict):
            return {"error": _BAD}
        return await self._answer(tool, args)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._handlers.add(task)
        try:
            if self._closed:
                return
            try:
                line: bytes | None = await reader.readuntil(b"\n")
            except asyncio.IncompleteReadError as exc:
                line = exc.partial
            except asyncio.LimitOverrunError:
                line = None
            reply = await self._reply(line)
            try:
                writer.write(
                    json.dumps(reply, ensure_ascii=False).encode("utf-8", "backslashreplace")
                    + b"\n"
                )
                if not self._closed:
                    await writer.drain()
            except ConnectionError:
                pass
        finally:
            writer.close()
            if task is not None:
                self._handlers.discard(task)

    async def close(self) -> None:
        if self._server is None:
            return
        self._closed = True
        self._server.close()
        self.path.unlink(missing_ok=True)
        handlers = list(self._handlers)
        for task in handlers:
            task.cancel()
        caller = asyncio.current_task()
        cancelled: asyncio.CancelledError | None = None
        hurried = False
        while pending := [task for task in handlers if not task.done()]:
            if not hurried and caller is not None and caller.cancelling() > 1:
                hurried = True
                for task in pending:
                    task.cancel()
            try:
                await asyncio.wait(pending)
            except asyncio.CancelledError as exc:
                cancelled = cancelled or exc
        self._server.abort_clients()
        try:
            await self._server.wait_closed()
        except asyncio.CancelledError as exc:
            cancelled = cancelled or exc
        if cancelled is not None:
            raise cancelled

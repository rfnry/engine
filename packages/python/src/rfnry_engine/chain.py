import asyncio
import hashlib
import json
from collections.abc import Callable, Sequence
from typing import Any

from rfnry_engine.models import Content, Message, Part, Text, ToolSpec

YIELD_EVERY = 100


def _part(part: Part) -> dict[str, Any]:
    if isinstance(part, Text):
        return part.model_dump()
    held = part.model_dump(exclude={"data"})
    held["data"] = None if part.data is None else hashlib.sha256(part.data).hexdigest()
    return held


def _content(content: Content) -> str | list[dict[str, Any]]:
    return content if isinstance(content, str) else [_part(part) for part in content]


def _canonical(said: Any) -> bytes:
    return json.dumps(said, sort_keys=True, ensure_ascii=False).encode()


def preload_digest(parts: list[Part]) -> str:
    return hashlib.sha256(_canonical([_part(part) for part in parts])).hexdigest()


def opening(model: str | None, system: str, tools: list[ToolSpec], preload: str) -> bytes:
    head = {
        "model": model,
        "system": system,
        "tools": [tool.model_dump(mode="json") for tool in tools],
        "preload": preload,
    }
    return hashlib.sha256(_canonical(head)).digest()


def extend(digest: bytes, message: Message) -> bytes:
    said = message.model_dump(mode="json", exclude={"prefix", "usage", "content"}, fallback=repr)
    said["content"] = _content(message.content)
    return hashlib.sha256(digest + _canonical(said)).digest()


def mark(digest: bytes) -> str:
    return digest.hex()[:16]


class Chain:
    def __init__(self, head: Callable[[], bytes]) -> None:
        self._head = head
        self._digest: bytes | None = None
        self._hashed = 0

    async def settle(self, history: list[Message]) -> list[Message]:
        given = [index for index, message in enumerate(history) if message.reasoning is not None]
        if not given:
            return history
        sent = list(history)
        for index in given:
            message = sent[index]
            if message.prefix != mark(await self._reach(sent, index)):
                sent[index] = message.model_copy(update={"reasoning": None})
        return sent

    async def stamp(self, messages: Sequence[Message]) -> str:
        return mark(await self._reach(messages, len(messages)))

    async def _reach(self, messages: Sequence[Message], end: int) -> bytes:
        if self._digest is None:
            self._digest = self._head()
        while self._hashed < end:
            self._digest = extend(self._digest, messages[self._hashed])
            self._hashed += 1
            if self._hashed % YIELD_EVERY == 0:
                await asyncio.sleep(0)
        return self._digest

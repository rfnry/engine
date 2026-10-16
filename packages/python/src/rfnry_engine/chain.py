from __future__ import annotations

import hashlib
import json

from rfnry_engine.models import Message, Part, ToolSpec


def opening(model: str | None, system: str, tools: list[ToolSpec], preload: list[Part]) -> bytes:
    head = {
        "model": model,
        "system": system,
        "tools": [tool.model_dump(mode="json") for tool in tools],
        "preload": [part.model_dump(mode="json") for part in preload],
    }
    return hashlib.sha256(json.dumps(head, sort_keys=True, ensure_ascii=False).encode()).digest()


def extend(digest: bytes, message: Message) -> bytes:
    said = message.model_dump(mode="json", exclude={"prefix", "usage"}, fallback=repr)
    return hashlib.sha256(
        digest + json.dumps(said, sort_keys=True, ensure_ascii=False).encode()
    ).digest()


def mark(digest: bytes) -> str:
    return digest.hex()[:16]


def settle(history: list[Message], digest: bytes) -> tuple[list[Message], bytes]:
    sent: list[Message] = []
    for message in history:
        if message.reasoning is not None and message.prefix != mark(digest):
            message = message.model_copy(update={"reasoning": None})
        sent.append(message)
        digest = extend(digest, message)
    return sent, digest

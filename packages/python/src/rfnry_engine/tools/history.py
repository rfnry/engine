from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from typing import Any

import regex
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from rfnry_engine.errors import InputError
from rfnry_engine.models import HistoryLike, Message, Part, Text
from rfnry_engine.tools.tools import Result, Tool

_PAGE = 100
_MESSAGES = TypeAdapter(list[Message])
_DESCRIPTION = (
    "Read the earlier messages of this conversation, the ones before what you were given, each with "
    "its position. Pass the first position you read as before to read further back. With pattern, "
    "only the messages whose text matches."
)

Found = list[tuple[int, Message]]


class HistoryArgs(BaseModel):
    before: int | None = Field(default=None, ge=0)
    limit: int = Field(default=20, ge=1, le=100)
    pattern: str | None = None
    case_insensitive: bool = False


async def _read(history: Sequence[object] | HistoryLike, first: int, stop: int) -> list[Message]:
    found = (
        await history.read(first, stop) if isinstance(history, HistoryLike) else history[first:stop]
    )
    try:
        return _MESSAGES.validate_python(list(found), from_attributes=True)
    except ValidationError as exc:
        raise InputError(
            f"history: the conversation does not fit the contract: {exc.errors(include_url=False)}"
        ) from exc


def _said(message: Message) -> str:
    content = message.content
    texts = (
        [content] if isinstance(content, str) else [p.text for p in content if isinstance(p, Text)]
    )
    calls = [
        json.dumps({"name": c.name, "args": c.args}, ensure_ascii=False) for c in message.tool_calls
    ]
    return "\n".join([*texts, *calls])


def _shown(position: int, message: Message) -> list[Part]:
    header: dict[str, Any] = {"position": position, "role": message.role}
    if message.tool_calls:
        header["tool_calls"] = [call.model_dump() for call in message.tool_calls]
    if message.role == "tool":
        header |= {"tool_call_id": message.tool_call_id, "is_error": message.is_error}
    parts: list[Part] = [Text(text=json.dumps(header, ensure_ascii=False))]
    if isinstance(message.content, str):
        return [*parts, Text(text=message.content)] if message.content else parts
    return [*parts, *message.content]


def _size(parts: list[Part]) -> int:
    return sum(len(part.text) for part in parts if isinstance(part, Text))


def _clip(parts: list[Part], chars: int) -> list[Part]:
    kept: list[Part] = []
    for part in parts:
        if isinstance(part, Text):
            kept.append(Text(text=part.text[: max(chars, 0)]))
            chars -= len(part.text)
        else:
            kept.append(part)
    return kept


def _page(found: Found, chars: int) -> tuple[list[Part], int]:
    shown: list[list[Part]] = []
    used = 0
    first = found[-1][0]
    for position, message in reversed(found):
        parts = _shown(position, message)
        size = _size(parts)
        if shown and used + size > chars:
            break
        if size > chars:
            parts = [
                *_clip(parts, chars),
                Text(text=f"[position {position} cut at {chars} characters]"),
            ]
        shown.append(parts)
        used += size
        first = position
    return [part for parts in reversed(shown) for part in parts], first


def _matching(pattern: regex.Pattern[str], indexed: Found, deadline: float | None) -> Found:
    hits: Found = []
    for position, message in indexed:
        left = None if deadline is None else max(deadline - time.monotonic(), 0.0)
        try:
            if pattern.search(_said(message), timeout=left) is not None:
                hits.append((position, message))
        except TimeoutError as exc:
            raise InputError(
                "history: the pattern ran past the tool's deadline; simplify it"
            ) from exc
    return hits


async def _reading(
    history: Sequence[object] | HistoryLike, before: int, limit: int, chars: int
) -> Result:
    first = max(0, before - limit)
    page = await _read(history, first, before)
    if not page:
        return "no earlier messages"
    parts, shown = _page(list(enumerate(page, first)), chars)
    if shown > 0:
        parts.append(Text(text=f"[more before position {shown}: pass before={shown} to read them]"))
    return parts


async def _searching(
    history: Sequence[object] | HistoryLike,
    before: int,
    args: HistoryArgs,
    walk: int,
    chars: int,
    timeout: float | None,
) -> Result:
    try:
        pattern = regex.compile(
            args.pattern or "", regex.IGNORECASE if args.case_insensitive else 0
        )
    except regex.error as exc:
        raise InputError(f"history: invalid pattern: {exc}") from exc
    deadline = None if timeout is None else time.monotonic() + timeout
    found: Found = []
    position, walked = before, 0
    while position > 0 and len(found) < args.limit and walked < walk:
        first = max(0, position - min(_PAGE, walk - walked))
        page = await _read(history, first, position)
        hits = await asyncio.to_thread(_matching, pattern, list(enumerate(page, first)), deadline)
        room = args.limit - len(found)
        found = hits[-room:] + found
        walked += position - first
        position = hits[-room][0] if len(hits) > room else first
    if not found:
        if position > 0:
            return (
                f"no matches between positions {position} and {before}: "
                f"pass before={position} to search further"
            )
        return f"no matches before position {before}"
    parts, shown = _page(found, chars)
    resume = shown if shown != found[0][0] else position
    if resume > 0:
        parts.append(
            Text(
                text=f"[searched back to position {resume}: pass before={resume} to search further]"
            )
        )
    return parts


def history_tool(
    history: Sequence[object] | HistoryLike,
    start: int,
    *,
    walk: int,
    chars: int,
    timeout: float | None,
) -> Tool:
    async def body(args: dict[str, Any]) -> Result:
        a = HistoryArgs.model_validate(args)
        before = start if a.before is None else min(a.before, start)
        if a.pattern is None:
            return await _reading(history, before, a.limit, chars)
        return await _searching(history, before, a, walk, chars, timeout)

    return Tool("History", _DESCRIPTION, body=body, args=HistoryArgs)

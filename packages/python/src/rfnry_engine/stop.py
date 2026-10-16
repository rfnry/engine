from __future__ import annotations

import asyncio
from collections.abc import Callable, Collection, Sequence
from typing import Any, Literal

Stopped = Literal["now", "after_step"]


class Stop:
    def __init__(self) -> None:
        self._mode: Stopped | None = None
        self._presses = 0
        self._pressed: asyncio.Future[None] | None = None

    def now(self) -> None:
        self._mode = "now"
        self._presses += 1
        if self._pressed is not None and not self._pressed.done():
            self._pressed.set_result(None)
        self._pressed = None

    def after_step(self) -> None:
        if self._mode is None:
            self._mode = "after_step"


def requested(stop: Stop | None) -> Stopped | None:
    return None if stop is None else stop._mode


def _press(stop: Stop, seen: int) -> asyncio.Future[None]:
    loop = asyncio.get_running_loop()
    if stop._presses > seen:
        pressed = loop.create_future()
        pressed.set_result(None)
        return pressed
    if stop._pressed is None:
        stop._pressed = loop.create_future()
    return stop._pressed


async def wait(
    tasks: Sequence[asyncio.Future[Any]],
    stop: Stop | None,
    spared: Callable[[], Collection[asyncio.Future[Any]]] = tuple,
) -> bool:
    seen = 0
    cancelled: asyncio.CancelledError | None = None
    while pending := [task for task in tasks if not task.done()]:
        watched = set(pending) if stop is None else {*pending, _press(stop, seen)}
        try:
            await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError as exc:
            cancelled = cancelled or exc
            for task in pending:
                task.cancel()
            continue
        if stop is not None and stop._presses > seen:
            seen += 1
            kept = spared()
            for task in pending:
                if task not in kept:
                    task.cancel()
    for task in tasks:
        if not task.cancelled():
            task.exception()
    if cancelled is not None:
        raise cancelled
    return seen > 0


async def cancel(tasks: Sequence[asyncio.Future[Any]]) -> None:
    for task in tasks:
        task.cancel()
    await join(tasks)


async def join(tasks: Sequence[asyncio.Future[Any]]) -> None:
    cancelled: asyncio.CancelledError | None = None
    while pending := [task for task in tasks if not task.done()]:
        try:
            await asyncio.wait(pending)
        except asyncio.CancelledError as exc:
            cancelled = cancelled or exc
    for task in tasks:
        if not task.cancelled():
            task.exception()
    if cancelled is not None:
        raise cancelled

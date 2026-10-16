from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, Literal

Stopped = Literal["now", "after_step"]


class Stop:
    def __init__(self) -> None:
        self._mode: Stopped | None = None
        self._now: asyncio.Future[None] | None = None

    def now(self) -> None:
        self._mode = "now"
        if self._now is not None and not self._now.done():
            self._now.set_result(None)

    def after_step(self) -> None:
        if self._mode is None:
            self._mode = "after_step"


def requested(stop: Stop | None) -> Stopped | None:
    return None if stop is None else stop._mode


def _stopping(stop: Stop) -> asyncio.Future[None]:
    if stop._now is None:
        stop._now = asyncio.get_running_loop().create_future()
        if stop._mode == "now":
            stop._now.set_result(None)
    return stop._now


async def wait(tasks: list[asyncio.Task[Any]], stop: Stop | None) -> bool:
    pending: set[asyncio.Future[Any]] = set(tasks)
    stopping = None if stop is None else _stopping(stop)
    try:
        while pending:
            watched = pending if stopping is None else pending | {stopping}
            done, _ = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
            pending -= done
            if stopping is not None and stopping in done:
                return True
        return False
    except BaseException:
        await cancel(tasks)
        raise


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

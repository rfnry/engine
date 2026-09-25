from __future__ import annotations

import asyncio
from typing import Any, Literal

Stopped = Literal["now", "after_step"]


class Stop:
    def __init__(self) -> None:
        self._mode: Stopped | None = None
        self._now = asyncio.Event()

    def now(self) -> None:
        self._mode = "now"
        self._now.set()

    def after_step(self) -> None:
        if self._mode is None:
            self._mode = "after_step"


def requested(stop: Stop | None) -> Stopped | None:
    return None if stop is None else stop._mode


async def wait(tasks: list[asyncio.Task[Any]], stop: Stop | None) -> bool:
    pending: set[asyncio.Future[Any]] = set(tasks)
    waiter = asyncio.ensure_future(stop._now.wait()) if stop is not None else None
    try:
        while pending:
            watched = pending if waiter is None else pending | {waiter}
            done, _ = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
            pending -= done
            if waiter is not None and waiter in done:
                return True
        return False
    except BaseException:
        for task in tasks:
            task.cancel()
        raise
    finally:
        if waiter is not None:
            waiter.cancel()


async def cancel(tasks: list[asyncio.Task[Any]]) -> None:
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.wait(tasks)
    for task in tasks:
        if not task.cancelled():
            task.exception()

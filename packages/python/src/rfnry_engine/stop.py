import asyncio
from collections.abc import Awaitable, Callable, Collection, Sequence
from typing import Any, Literal

Stopped = Literal["now", "after_step"]


class Stop:
    def __init__(self) -> None:
        self._mode: Stopped | None = None
        self._presses = 0
        self._pressed: asyncio.Future[None] | None = None
        self._parent: Stop | None = None
        self._children: set[Stop] = set()
        self._cut: Callable[[], None] | None = None
        self._routed: Stop | None = None

    def now(self) -> None:
        self._mode = "now"
        self._presses += 1
        if self._pressed is not None and not self._pressed.done():
            self._pressed.set_result(None)
        self._pressed = None
        for child in self._children:
            child.now()

    def after_step(self) -> None:
        if self._mode is None:
            self._mode = "after_step"
        for child in self._children:
            child.after_step()

    def _below(self) -> Stop:
        child = Stop()
        child._parent = self
        self._children.add(child)
        if self._mode == "now":
            child.now()
        elif self._mode == "after_step":
            child.after_step()
        return child

    def _drop(self) -> None:
        if self._parent is not None:
            self._parent._children.discard(self)

    def _from(self, other: Stop | None) -> bool:
        stop: Stop | None = self
        while stop is not None:
            if stop is other:
                return True
            stop = stop._parent
        return False


def requested(stop: Stop | None) -> Stopped | None:
    return None if stop is None else stop._mode


def _ready() -> asyncio.Future[None]:
    ready = asyncio.get_running_loop().create_future()
    ready.set_result(None)
    return ready


def _press(stop: Stop, seen: int) -> asyncio.Future[None]:
    if stop._presses > seen:
        return _ready()
    if stop._pressed is None:
        stop._pressed = asyncio.get_running_loop().create_future()
    return stop._pressed


async def wait(
    tasks: Sequence[asyncio.Future[Any]],
    stop: Stop | None,
    spared: Callable[[], Collection[asyncio.Future[Any]]] = tuple,
    calm: int | None = None,
) -> bool:
    runner = asyncio.current_task()
    if calm is None:
        calm = 0 if runner is None else runner.cancelling()
    seen = taken = 0
    cancelled: asyncio.CancelledError | None = None

    def owed() -> bool:
        return runner is not None and runner.cancelling() - calm > taken

    while pending := [task for task in tasks if not task.done()]:
        watched = set(pending)
        if stop is not None:
            watched.add(_press(stop, seen))
        if owed():
            watched.add(_ready())
        try:
            await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError as exc:
            cancelled = cancelled or exc
            if owed():
                taken += 1
                for task in pending:
                    task.cancel()
            continue
        if owed():
            taken += 1
            for task in pending:
                task.cancel()
        elif stop is not None and stop._presses > seen:
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


async def unless_cut[T](stop: Stop, work: Awaitable[T]) -> T | None:
    task = asyncio.current_task()
    if task is None:
        return await work
    owner: asyncio.Task[Any] = task
    calm = owner.cancelling()
    fired = False

    def cut() -> None:
        nonlocal fired
        if not fired:
            fired = True
            owner.cancel()

    stop._cut = cut
    try:
        done = await work
    except asyncio.CancelledError:
        if fired and owner.uncancel() <= calm:
            return None
        raise
    finally:
        stop._cut = None
    if fired:
        owner.uncancel()
    return done


async def cancel(tasks: Sequence[asyncio.Future[Any]]) -> None:
    for task in tasks:
        task.cancel()
    await wait(tasks, None, calm=1)


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

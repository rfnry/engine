import asyncio
import contextvars
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from rfnry_engine.records import CallInfo, Decision, EventHandler, Generation, Run
from rfnry_engine.stop import Stop


@dataclass
class Foreground:
    stop: Stop | None
    on_event: EventHandler | None
    task: asyncio.Future[Any] | None = None
    stopped: bool = False
    live: int = 0
    pressed: bool = False
    over: bool = False

    def settle(self) -> None:
        if self.pressed and self.live == 0 and not self.over and self.task is not None:
            self.pressed = False
            self.task.cancel()


FOREGROUND: contextvars.ContextVar[Foreground | None] = contextvars.ContextVar(
    "foreground", default=None
)


def current() -> Foreground | None:
    here = FOREGROUND.get()
    return None if here is None or here.over else here


@contextmanager
def taking(stop: Stop | None) -> Iterator[None]:
    here = current()
    if stop is None or here is None or here.stop is not stop:
        yield
        return
    here.live += 1
    try:
        yield
    finally:
        here.live -= 1
        if here.live == 0 and here.pressed:
            asyncio.get_running_loop().call_soon(here.settle)


def halted(run: Run) -> bool:
    while isinstance(run, Decision) and run.routed_to is not None:
        run = run.routed_to
    return isinstance(run, Generation) and run.stopped is not None


def hooks(
    on_event: EventHandler | None, stop: Callable[[], Stop | None] | None
) -> tuple[EventHandler | None, Stop | None]:
    here = current()
    return (
        on_event if on_event is not None or here is None else here.on_event,
        stop() if stop is not None else None if here is None else here.stop,
    )


def returned(run: Run) -> Run:
    here = current()
    if here is not None and halted(run):
        here.stopped = True
    return run


CALL: contextvars.ContextVar[CallInfo | None] = contextvars.ContextVar("call", default=None)
RECORDING: contextvars.ContextVar[bool] = contextvars.ContextVar("recording", default=True)


def current_call() -> CallInfo | None:
    return CALL.get()

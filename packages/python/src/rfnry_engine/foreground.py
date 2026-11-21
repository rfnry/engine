import asyncio
import contextvars
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from rfnry_engine.records import CallInfo, Decision, EventHandler, Generation, Pipe, Run
from rfnry_engine.stop import Stop


@dataclass(frozen=True)
class Lent:
    env: dict[str, str]
    said: Callable[[Pipe, str], Awaitable[None]] | None = None
    call: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]] | None = None


@dataclass
class Foreground:
    stop: Stop | None
    on_event: EventHandler | None
    task: asyncio.Future[Any] | None = None
    stopped: bool = False
    live: int = 0
    pressed: bool = False
    over: bool = False
    lent: Lent | None = None
    failed: Exception | None = None
    result: Any = None
    approved: bool | None = None

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
    if stop is None or here is None or not stop._from(here.stop):
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


@contextmanager
def hooks(
    on_event: EventHandler | None, stop: Callable[[], Stop | None] | None
) -> Iterator[tuple[EventHandler | None, Stop | None]]:
    here = current()
    above = None if here is None else here.stop
    given = above if stop is None else stop()
    if above is not None and above._from(given):
        given = above
    held = None if given is None else given._below()
    try:
        yield on_event if on_event is not None or here is None else here.on_event, held
    finally:
        if held is not None:
            held._drop()


def returned(run: Run) -> Run:
    if halted(run):
        answer_stopped()
    return run


def answer_stopped() -> None:
    here = current()
    if here is not None:
        here.stopped = True


def answers_stopped() -> bool:
    here = current()
    return here is not None and here.stopped


CALL: contextvars.ContextVar[CallInfo | None] = contextvars.ContextVar("call", default=None)
RECORDING: contextvars.ContextVar[bool] = contextvars.ContextVar("recording", default=True)


def current_call() -> CallInfo | None:
    return CALL.get()

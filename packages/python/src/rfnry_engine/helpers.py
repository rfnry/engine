import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from rfnry_engine.errors import error_text
from rfnry_engine.foreground import halted
from rfnry_engine.models import Message, Notice
from rfnry_engine.records import Decision, Generation
from rfnry_engine.stop import Stop, pressed, requested
from rfnry_engine.tools.delegate import DELEGATE_TOOL, child_result

Ending = Literal["finished", "failed", "stopped"]
JOB = re.compile(r"d([1-9][0-9]*)")


def notice_text(job: str, agent: str, kind: str, body: str) -> str:
    return f"[helper {job} · {agent} · {kind}]\n{body}"


def _number(job: object) -> int:
    found = JOB.fullmatch(job) if isinstance(job, str) else None
    return 0 if found is None else int(found[1])


def _handed(message: Message) -> int:
    if not isinstance(message.content, str):
        return 0
    try:
        held = json.loads(message.content)
    except ValueError:
        return 0
    return _number(held.get("job") if isinstance(held, dict) else None)


def first_job(history: Sequence[Message]) -> int:
    delegated = {
        call.id for message in history for call in message.tool_calls if call.name == DELEGATE_TOOL
    }
    handed = [
        _handed(message)
        for message in history
        if message.role == "tool" and message.tool_call_id in delegated
    ]
    noticed = [_number(message.notice.job) for message in history if message.notice is not None]
    return max([*handed, *noticed], default=0) + 1


def _ending(record: Generation | Decision | None) -> tuple[Ending, str]:
    if record is None:
        return "stopped", ""
    if record.error is not None:
        return "failed", record.error
    if halted(record):
        return "stopped", child_result(record)
    return "finished", child_result(record)


@dataclass
class Helper:
    job: str
    agent: str
    run_id: str
    stop: Stop
    inbox: asyncio.Queue[Any]
    started: datetime
    task: asyncio.Task[Any] | None


class Helpers:
    def __init__(
        self, lead: str, cap: int | None, first: int, children: list[Generation | Decision]
    ) -> None:
        self.lead = lead
        self.cap = cap
        self.children = children
        self.ready = asyncio.Event()
        self._next = first
        self._running: dict[str, Helper] = {}
        self._notices: list[Message] = []

    def start(
        self,
        agent: str,
        run_id: str,
        stop: Stop,
        run: Callable[[Helper], Awaitable[Generation | Decision | None]],
    ) -> Helper:
        helper = Helper(
            job=f"d{self._next}",
            agent=agent,
            run_id=run_id,
            stop=stop,
            inbox=asyncio.Queue(),
            started=datetime.now(UTC),
            task=None,
        )
        self._next += 1
        self._running[helper.job] = helper
        helper.task = asyncio.ensure_future(self._ran(helper, run))
        return helper

    async def _ran(
        self, helper: Helper, run: Callable[[Helper], Awaitable[Generation | Decision | None]]
    ) -> None:
        try:
            record = await run(helper)
        except Exception as exc:
            self._ended(helper, None, "failed", error_text(exc))
            return
        self._ended(helper, record, *_ending(record))

    def _ended(
        self, helper: Helper, record: Generation | Decision | None, kind: Ending, body: str
    ) -> None:
        del self._running[helper.job]
        if record is not None:
            self.children.append(record)
        self._notices.append(
            Message(
                role="user",
                content=notice_text(helper.job, helper.agent, kind, body),
                notice=Notice(job=helper.job, agent=helper.agent, run_id=helper.run_id, kind=kind),
            )
        )
        self.ready.set()

    def running(self) -> list[Helper]:
        return list(self._running.values())

    def waiting(self) -> bool:
        return bool(self._notices)

    def take(self) -> list[Message]:
        taken, self._notices = self._notices, []
        self.ready.clear()
        return taken

    async def wake(self, inbox: asyncio.Queue[Any], stop: Stop) -> list[Any]:
        if self.ready.is_set() or requested(stop) is not None:
            return []
        ready = asyncio.ensure_future(self.ready.wait())
        getting = asyncio.ensure_future(inbox.get())
        try:
            await asyncio.wait([ready, getting, pressed(stop)], return_when=asyncio.FIRST_COMPLETED)
        finally:
            ready.cancel()
            getting.cancel()
        return [getting.result()] if getting.done() and not getting.cancelled() else []

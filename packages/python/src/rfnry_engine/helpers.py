import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal, cast

from rfnry_engine.errors import error_text
from rfnry_engine.foreground import FOREGROUND, Foreground, halted
from rfnry_engine.manifest import Kind
from rfnry_engine.models import Message, Notice
from rfnry_engine.records import Decision, Generation
from rfnry_engine.stop import Stop, Stopped, asked, cut_through, press, pressed, requested, wait
from rfnry_engine.tools.delegate import DELEGATE_TOOL, child_result, run_result
from rfnry_engine.tools.workspace import JOB_TOOL, JOBS_TOOL

Ending = Literal["finished", "failed", "stopped"]
JOB = re.compile(r"d([1-9][0-9]*)")


class Unanswered(Enum):
    ENDED = "ended"
    STOPPED = "stopped"


def notice_text(job: str, agent: str, kind: str, body: str) -> str:
    return f"[helper {job} · {agent} · {kind}]\n{body}"


def _number(job: object) -> int:
    found = JOB.fullmatch(job) if isinstance(job, str) else None
    return 0 if found is None else int(found[1])


def _shown(message: Message) -> list[int]:
    if not isinstance(message.content, str):
        return []
    try:
        held = json.loads(message.content)
    except ValueError:
        return []
    if not isinstance(held, dict):
        return []
    listed = held.get("jobs")
    entries = listed if isinstance(listed, list) else []
    return [
        _number(held.get("job")),
        *(_number(entry.get("job")) for entry in entries if isinstance(entry, dict)),
    ]


def first_job(history: Sequence[Message]) -> int:
    calls = [call for message in history for call in message.tool_calls]
    answered = {call.id for call in calls if call.name in (DELEGATE_TOOL, JOB_TOOL, JOBS_TOOL)}
    named = [_number(call.args.get("id")) for call in calls if call.name == JOB_TOOL]
    shown = [
        number
        for message in history
        if message.role == "tool" and message.tool_call_id in answered
        for number in _shown(message)
    ]
    noticed = [_number(message.notice.job) for message in history if message.notice is not None]
    return max([*named, *shown, *noticed], default=0) + 1


async def _arrival(inbox: asyncio.Queue[Any]) -> Any:
    try:
        return await inbox.get()
    except asyncio.QueueShutDown:
        return await asyncio.get_running_loop().create_future()


def given_back(inbox: asyncio.Queue[Any], woken: list[Any]) -> None:
    while woken:
        with suppress(asyncio.QueueShutDown, asyncio.QueueFull):
            inbox.put_nowait(woken.pop(0))
        inbox.task_done()


def _ending(record: Generation | Decision | None, cut: bool) -> tuple[Ending, str]:
    if record is None:
        return "stopped", ""
    if record.error is not None:
        return "failed", record.error
    if cut or halted(record):
        return "stopped", child_result(record)
    return "finished", child_result(record)


@dataclass(eq=False)
class Asking:
    question: str
    answer: asyncio.Future[str | Unanswered]
    given: bool = False


@dataclass
class Helper:
    job: str
    agent: str
    kind: Kind
    run_id: str
    stop: Stop
    inbox: asyncio.Queue[Any]
    started: datetime
    task: asyncio.Task[Any] | None
    ended: Ending | None = None
    said: str = ""
    record: Generation | Decision | None = None
    asks: list[Asking] = field(default_factory=list)

    def pending(self) -> list[Asking]:
        return [asking for asking in self.asks if not asking.answer.done()]


class Helpers:
    def __init__(
        self, lead: str, cap: int | None, first: int, children: list[Generation | Decision]
    ) -> None:
        self.lead = lead
        self.cap = cap
        self.children = children
        self.ready = asyncio.Event()
        self._next = first
        self._helpers: dict[str, Helper] = {}
        self._notices: list[Message] = []
        self._settling = False

    def start(
        self,
        agent: str,
        kind: Kind,
        run_id: str,
        stop: Stop,
        run: Callable[[Helper], Awaitable[Generation | Decision | None]],
    ) -> Helper:
        helper = Helper(
            job=f"d{self._next}",
            agent=agent,
            kind=kind,
            run_id=run_id,
            stop=stop,
            inbox=asyncio.Queue(),
            started=datetime.now(UTC),
            task=None,
        )
        self._next += 1
        self._helpers[helper.job] = helper
        helper.task = asyncio.ensure_future(self._ran(helper, run))
        return helper

    async def _ran(
        self, helper: Helper, run: Callable[[Helper], Awaitable[Generation | Decision | None]]
    ) -> None:
        here = Foreground(None, None)
        FOREGROUND.set(here)
        try:
            record = await run(helper)
        except Exception as exc:
            self._ended(helper, None, "failed", error_text(exc))
            return
        self._ended(helper, record, *_ending(record, here.stopped))

    def _ended(
        self, helper: Helper, record: Generation | Decision | None, kind: Ending, body: str
    ) -> None:
        helper.ended, helper.said, helper.record = kind, body, record
        if record is not None:
            self.children.append(record)
        self._noticed(helper, kind, body)

    def _noticed(self, helper: Helper, kind: Ending | Literal["asks"], body: str) -> None:
        self._notices.append(
            Message(
                role="user",
                content=notice_text(helper.job, helper.agent, kind, body),
                notice=Notice(job=helper.job, agent=helper.agent, run_id=helper.run_id, kind=kind),
            )
        )
        self.ready.set()

    async def ask(self, helper: Helper, question: str) -> str | Unanswered:
        if self._settling:
            return Unanswered.STOPPED
        asking = Asking(question, asyncio.get_running_loop().create_future())
        helper.asks.append(asking)
        self._noticed(helper, "asks", question)
        return await asking.answer

    def fail_given_asks(self) -> None:
        for helper in self._helpers.values():
            for asking in helper.pending():
                if asking.given:
                    asking.answer.set_result(Unanswered.ENDED)

    def running(self) -> list[Helper]:
        return [helper for helper in self._helpers.values() if helper.ended is None]

    def get(self, job: str) -> Helper | None:
        return self._helpers.get(job)

    def listed(self) -> list[dict[str, Any]]:
        return [
            {
                "job": helper.job,
                "agent": helper.agent,
                "started": helper.started.isoformat(timespec="seconds"),
                "running": helper.ended is None,
                **({} if helper.ended is None else {"ended": helper.ended}),
            }
            for helper in self._helpers.values()
        ]

    def read(self, job: str) -> dict[str, Any]:
        helper = self._helpers[job]
        question = next((asking.question for asking in helper.pending()), None)
        held = {"running": helper.ended is None, "agent": helper.agent, "asking": question}
        if helper.ended == "finished":
            result = run_result(helper.record)
            return {**held, **({"reply": result} if isinstance(result, str) else result)}
        if helper.ended is not None:
            return {**held, ("error" if helper.ended == "failed" else "stopped"): helper.said}
        return held

    async def wait(self, job: str, seconds: float | None) -> None:
        helper = self._helpers[job]
        if helper.ended is not None or helper.pending() or seconds == 0:
            return
        with suppress(TimeoutError):
            async with asyncio.timeout(seconds):
                await self.ready.wait()

    async def stop_one(self, job: str, seconds: float | None) -> None:
        helper = self._helpers[job]
        press(helper.stop)
        await asyncio.wait([cast(asyncio.Task[Any], helper.task)], timeout=seconds)

    async def send(self, job: str, text: str) -> None:
        helper = self._helpers[job]
        pending = helper.pending()
        if pending:
            pending[0].answer.set_result(text)
            return
        await helper.inbox.put(text)

    def waiting(self) -> bool:
        return bool(self._notices)

    def take(self) -> list[Message]:
        taken, self._notices = self._notices, []
        self.ready.clear()
        for helper in self._helpers.values():
            for asking in helper.pending():
                asking.given = True
        return taken

    async def wake(self, inbox: asyncio.Queue[Any], stop: Stop) -> list[Any]:
        if self.ready.is_set() or requested(stop) is not None:
            return []
        if not inbox.empty():
            return [inbox.get_nowait()]
        loop = asyncio.get_running_loop()
        ready = asyncio.ensure_future(self.ready.wait())
        told = asked(stop)
        holding = True
        try:
            async with asyncio.timeout(None) as held:

                def woke(_: object) -> None:
                    nonlocal holding
                    if holding:
                        holding = False
                        held.reschedule(loop.time())

                ready.add_done_callback(woke)
                told.add_done_callback(woke)
                try:
                    first = await _arrival(inbox)
                finally:
                    holding = False
                    ready.remove_done_callback(woke)
                    told.remove_done_callback(woke)
                    ready.cancel()
        except TimeoutError:
            return []
        return [first]

    async def settle(self, mode: Stopped, stop: Stop) -> None:
        self._settling = True
        for helper in self.running():
            for asking in helper.pending():
                asking.answer.set_result(Unanswered.STOPPED)
        tasks = [helper.task for helper in self.running() if helper.task is not None]
        if mode == "after_step":
            for helper in self.running():
                helper.stop.after_step()
            press = pressed(stop)
            while not press.done() and (left := [task for task in tasks if not task.done()]):
                await asyncio.wait([*left, press], return_when=asyncio.FIRST_COMPLETED)
        for helper in self.running():
            if requested(helper.stop) != "now":
                helper.stop.now()
            cut_through(helper.stop)
        if running := [task for task in tasks if not task.done()]:
            await asyncio.wait(running)

    async def cancel(self) -> None:
        tasks = [helper.task for helper in self.running() if helper.task is not None]
        for task in tasks:
            task.cancel()
        await wait(tasks, None)

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Collection, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import ValidationError

from rfnry_engine.conversation import live
from rfnry_engine.errors import _Unpausable, error_text
from rfnry_engine.foreground import FOREGROUND, Foreground, halted
from rfnry_engine.manifest import Kind
from rfnry_engine.models import Message, Notice, Pause, ToolCall
from rfnry_engine.records import Decision, Generation
from rfnry_engine.stop import Stop, Stopped, asked, cut_through, press, pressed, requested, wait
from rfnry_engine.tools.delegate import DELEGATE_TOOL, BackgroundArgs, child_result, run_result
from rfnry_engine.tools.workspace import JOB_TOOL, JOBS_TOOL

Ending = Literal["finished", "failed", "stopped", "paused"]
OUTCOMES = frozenset({"finished", "failed", "stopped", "lost"})
HELPER_ID = re.compile(r"d([0-9]+)")
ON_ITS_QUESTION = "waits for your answer to its question"


class Unanswered(Enum):
    ENDED = "ended"
    STOPPED = "stopped"
    PAUSED = "paused"


def notice_text(job: str, agent: str, kind: str, body: str) -> str:
    return f"[helper {job} · {agent} · {kind}]\n{body}"


def _notice(
    job: str,
    agent: str,
    run_id: str | None,
    kind: Ending | Literal["asks", "lost"],
    body: str,
) -> Message:
    return Message(
        role="user",
        content=notice_text(job, agent, kind, body),
        notice=Notice(job=job, agent=agent, run_id=run_id, kind=kind),
    )


def _number(job: object) -> int:
    found = HELPER_ID.fullmatch(job) if isinstance(job, str) else None
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


def _background(call: ToolCall) -> str | None:
    try:
        held = BackgroundArgs.model_validate(call.args)
    except ValidationError:
        return None
    return held.agent if held.background else None


def _started(message: Message) -> str | None:
    if message.is_error or not isinstance(message.content, str):
        return None
    try:
        held = json.loads(message.content)
    except ValueError:
        return None
    job = held.get("job") if isinstance(held, dict) and len(held) == 1 else None
    return job if isinstance(job, str) and _number(job) else None


def lost_notices(history: Sequence[Message], key: str, nameless: Collection[str]) -> list[Message]:
    calls = {
        call.id: agent
        for message in history
        for call in message.tool_calls
        if call.name == DELEGATE_TOOL and (agent := _background(call)) is not None
    }
    jobs = {
        job: agent
        for message in history
        if message.role == "tool"
        and (agent := calls.get(message.tool_call_id or "")) is not None
        and (job := _started(message)) is not None
    }
    paused = live(history)
    ended = {
        message.notice.job
        for message in [*history, *([] if paused is None else paused.held)]
        if message.notice is not None and message.notice.kind in OUTCOMES
    }
    unknown = (
        "the run that started it ended before its outcome reached you, and what it did is unknown"
    )
    return [
        _notice(
            job,
            jobs[job],
            None,
            "lost",
            unknown if jobs[job] in nameless else f"{unknown}; its conversation is {key}/{job}",
        )
        for job in sorted(jobs.keys() - ended, key=_number)
    ]


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


def _mark(record: Generation | Decision) -> Pause | None:
    run: Generation | Decision | None = record
    while isinstance(run, Decision):
        run = run.routed_to
    return None if run is None else live(run.conversation)


def _ending(record: Generation | Decision | None, cut: bool) -> tuple[Ending, str]:
    if record is None:
        return "stopped", ""
    if record.error is not None:
        return "failed", record.error
    if cut or halted(record):
        return "stopped", child_result(record)
    if record.paused:
        return "paused", f"waits for an answer to: {', '.join(w.call.name for w in record.paused)}"
    if _mark(record) is not None:
        return "paused", ON_ITS_QUESTION
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
    input: str
    kind: Kind
    run_id: str
    stop: Stop
    inbox: asyncio.Queue[Any]
    started: datetime
    again: Callable[[Helper], None]
    ended: Ending | None = None
    said: str = ""
    record: Generation | Decision | None = None
    asks: list[Asking] = field(default_factory=list)
    question: str | None = None
    over: asyncio.Future[None] = field(init=False)
    task: asyncio.Task[None] = field(init=False)

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
        self.fatal: _Unpausable | None = None
        self.pausing = False
        self._next = first
        self._helpers: dict[str, Helper] = {}
        self._notices: list[Message] = []
        self._keeping: set[str] = set()
        self._settling = False

    def upcoming(self) -> str:
        return f"d{self._next}"

    def skip(self) -> None:
        self._next += 1

    def start(
        self,
        agent: str,
        kind: Kind,
        run_id: str,
        stop: Stop,
        input: str,
        again: Callable[[Helper], None],
    ) -> Helper:
        helper = Helper(
            job=self.upcoming(),
            agent=agent,
            input=input,
            kind=kind,
            run_id=run_id,
            stop=stop,
            inbox=asyncio.Queue(),
            started=datetime.now(UTC),
            again=again,
        )
        self.skip()
        self._helpers[helper.job] = helper
        return helper

    def run(
        self,
        helper: Helper,
        run: Callable[[Helper], Awaitable[Generation | Decision | None]],
        keep: Callable[[Generation | Decision], Awaitable[None]],
    ) -> None:
        helper.ended, helper.said, helper.record, helper.question = None, "", None, None
        helper.over = asyncio.get_running_loop().create_future()
        helper.task = asyncio.ensure_future(self._ran(helper, run, keep))

    async def _ran(
        self,
        helper: Helper,
        run: Callable[[Helper], Awaitable[Generation | Decision | None]],
        keep: Callable[[Generation | Decision], Awaitable[None]],
    ) -> None:
        here = Foreground(None, None)
        FOREGROUND.set(here)
        try:
            record = await run(helper)
        except _Unpausable as exc:
            self._ended(helper, None, "failed", error_text(exc))
            self.fatal = self.fatal or exc
            self.ready.set()
            return
        except Exception as exc:
            said = error_text(exc)
            self._ended(helper, None, "failed", said)
            self._noticed(helper, "failed", said)
            return
        kind, body = _ending(record, here.stopped)
        self._ended(helper, record, kind, body)
        if record is not None:
            self._keeping.add(helper.job)
            try:
                await keep(record)
            finally:
                self._keeping.discard(helper.job)
            self.children.append(record)
        self._noticed(helper, kind, body)

    def _ended(
        self, helper: Helper, record: Generation | Decision | None, kind: Ending, body: str
    ) -> None:
        helper.ended, helper.said, helper.record = kind, body, record
        helper.over.set_result(None)

    def _noticed(self, helper: Helper, kind: Ending | Literal["asks"], body: str) -> None:
        self._notices.append(_notice(helper.job, helper.agent, helper.run_id, kind, body))
        self.ready.set()

    def _held_back(self, helper: Helper, question: str) -> Unanswered:
        if helper.question is not None:
            return Unanswered.ENDED
        helper.question = question
        return Unanswered.PAUSED

    async def ask(self, helper: Helper, question: str) -> str | Unanswered:
        if self._settling:
            return Unanswered.STOPPED
        if self.pausing:
            held = self._held_back(helper, question)
            if held is Unanswered.PAUSED:
                self._noticed(helper, "asks", question)
            return held
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

    def paused(self) -> list[Helper]:
        return [helper for helper in self._helpers.values() if helper.ended == "paused"]

    async def until_quiet(self, stop: Stop) -> None:
        if requested(stop) is not None:
            return
        self.pausing = True
        for helper in self.running():
            for asking in helper.pending():
                asking.answer.set_result(self._held_back(helper, asking.question))
        told = asked(stop)
        while requested(stop) is None and (
            left := [helper.over for helper in self.running() if not helper.task.done()]
        ):
            await asyncio.wait([*left, told], return_when=asyncio.FIRST_COMPLETED)

    def get(self, job: str) -> Helper | None:
        return self._helpers.get(job)

    def listed(self) -> list[dict[str, Any]]:
        return [
            {
                "job": helper.job,
                "agent": helper.agent,
                "input": helper.input,
                "started": helper.started.isoformat(timespec="seconds"),
                "running": helper.ended is None,
                **({} if helper.ended is None else {"ended": helper.ended}),
            }
            for helper in self._helpers.values()
        ]

    def read(self, job: str) -> dict[str, Any]:
        helper = self._helpers[job]
        question = (
            helper.question
            if helper.ended == "paused"
            else next((asking.question for asking in helper.pending()), None)
        )
        held = {
            "running": helper.ended is None,
            "agent": helper.agent,
            "input": helper.input,
            "asking": question,
        }
        if helper.ended == "finished":
            result = run_result(helper.record)
            return {**held, **({"reply": result} if isinstance(result, str) else result)}
        if helper.ended is not None:
            shown = {"failed": "error", "stopped": "stopped", "paused": "paused"}[helper.ended]
            return {**held, shown: helper.said}
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
        loop = asyncio.get_running_loop()
        until = None if seconds is None else loop.time() + seconds
        if helper.ended == "paused":
            await asyncio.wait([helper.task])
        if helper.ended == "paused":
            helper.again(helper)
        else:
            press(helper.stop)
        await asyncio.wait(
            [helper.over, helper.task],
            timeout=None if until is None else max(0.0, until - loop.time()),
            return_when=asyncio.FIRST_COMPLETED,
        )

    async def send(self, job: str, text: str) -> None:
        helper = self._helpers[job]
        pending = helper.pending()
        if pending:
            pending[0].answer.set_result(text)
            return
        await helper.inbox.put(text)

    def waiting(self) -> bool:
        return bool(self._notices or self._keeping or self.fatal)

    def _unfinished(self) -> list[asyncio.Task[None]]:
        return [helper.task for helper in self._helpers.values() if not helper.task.done()]

    async def settled(self) -> None:
        if kept := [self._helpers[job].task for job in self._keeping]:
            await asyncio.wait(kept)

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
        tasks = self._unfinished()
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
        while True:
            for helper in self.paused():
                if helper.task.done():
                    helper.again(helper)
            if not (running := self._unfinished()):
                return
            await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)

    async def cancel(self) -> None:
        tasks = self._unfinished()
        for task in tasks:
            task.cancel()
        await wait(tasks, None)

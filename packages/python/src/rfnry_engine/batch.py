import asyncio
import logging
import time
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from itertools import count
from typing import TYPE_CHECKING, Any

from rfnry_engine.errors import InputError, error_text
from rfnry_engine.foreground import Foreground, Lent
from rfnry_engine.models import Message, Refused, ToolCall, Waiting
from rfnry_engine.records import CommandOutput, EventHandler, Pipe, Step
from rfnry_engine.stop import Stop, requested, wait
from rfnry_engine.tools.tools import Tools
from rfnry_engine.transcript import Transcript
from rfnry_engine.tree import OUTPUT_TOOL

if TYPE_CHECKING:
    from rfnry_engine.engine import _ToolCalls

STOPPED = "stopped by the user"

logger = logging.getLogger("rfnry_engine.engine")


def _now() -> datetime:
    return datetime.now(UTC)


def _refusal(name: str, answer: bool | Refused) -> str:
    said = f"NotApproved: {name} was not approved"
    reason = answer.reason.strip() if isinstance(answer, Refused) else ""
    return f"{said}: {reason}" if reason else said


def _raise_first(failures: Sequence[BaseException]) -> None:
    for error in failures[1:]:
        logger.error("engine: a further call of the same batch failed: %s", error, exc_info=error)
    if failures:
        raise failures[0]


def unrun(call: ToolCall, why: str) -> Step:
    return Step(
        tool=call.name,
        args=call.args,
        result=f"NotRun: {call.name} was not run; {why}",
        is_error=True,
        at=_now(),
        duration_ms=0,
    )


class Batch:
    def __init__(
        self,
        calls: _ToolCalls,
        transcript: Transcript,
        steps: list[Step],
        adopt: Callable[[str | None], None],
    ) -> None:
        self.calls = calls
        self.transcript = transcript
        self.steps = steps
        self.adopt = adopt
        self.asked: list[ToolCall] = []

    def answer(self, call: ToolCall, step: Step) -> None:
        if not step.waiting:
            self.transcript.messages.append(
                Message(
                    role="tool",
                    tool_call_id=call.id,
                    content=step.result,
                    is_error=step.is_error,
                )
            )
        self.steps.append(step)
        self.adopt(step.child_run_id)

    def not_run(self, call: ToolCall, why: str) -> None:
        self.answer(call, unrun(call, why))

    def unwait(self, answered: Callable[[ToolCall, Step], Step]) -> list[tuple[ToolCall, Step]]:
        held = [(call, step) for call, step in self.calls.ran if step.waiting]
        self.calls.waiting.clear()
        self.calls.delegated.clear()
        self.calls.asking.clear()
        self.calls.ran[:] = [(call, step) for call, step in self.calls.ran if not step.waiting]
        given = []
        for call, step in held:
            made = answered(call, step)
            self.steps[next(n for n, kept in enumerate(self.steps) if kept is step)] = made
            self.transcript.messages.append(
                Message(role="tool", tool_call_id=call.id, content=made.result, is_error=True)
            )
            given.append((call, made))
        return given

    async def answer_all(
        self, batch: list[ToolCall], tasks: Sequence[asyncio.Task[Step] | None], began: float
    ) -> None:
        raised: list[BaseException] = []
        held: list[Exception] = []
        for call, task in zip(batch, tasks, strict=True):
            try:
                if task is None or task.cancelled():
                    await self.calls.cut(call, began)
                elif (failed := task.exception()) is not None:
                    raised.append(failed)
            except Exception as exc:
                raised.append(exc)
            built = self.built(call)
            if built is not None:
                self.answer(call, built)
            here = self.calls.foregrounds.pop(id(call), None)
            if here is not None and here.failed is not None and here.failed not in raised:
                held.append(here.failed)
        failures = [*raised, *held]
        lost = self.transcript.lost
        if lost is not None and any(error is lost for error in failures):
            failures = [lost, *(error for error in failures if error is not lost)]
        _raise_first(failures)

    async def run(self, batch: list[ToolCall], stop: Stop, stored: bool = True) -> bool:
        began = time.monotonic()
        if requested(stop) == "now":
            await self.answer_all(batch, [None] * len(batch), began)
            if stored:
                await self.transcript.save()
            return True
        tasks = [asyncio.ensure_future(self.calls.run(c)) for c in batch]

        def finishing() -> list[asyncio.Task[Step]]:
            kept = []
            for call, task in zip(batch, tasks, strict=True):
                here = self.calls.foregrounds.get(id(call))
                if here is not None and here.live > 0:
                    here.pressed = True
                    kept.append(task)
                elif id(call) in self.calls.approving or any(
                    call is done for done, _ in self.calls.ran
                ):
                    kept.append(task)
            return kept

        cut_short = await wait(tasks, stop, finishing)
        await self.answer_all(batch, tasks, began)
        if stored:
            await self.transcript.save()
        return cut_short

    def waiting(self) -> list[Waiting]:
        return [
            self.calls.waiting[id(call)] for call in self.asked if id(call) in self.calls.waiting
        ]

    def waits(self) -> bool:
        return bool(self.calls.waiting or self.calls.delegated or self.calls.asking)

    def built(self, call: ToolCall) -> Step | None:
        return next((step for done, step in self.calls.ran if done is call), None)

    def pending(self) -> dict[str, tuple[str, list[Waiting]]]:
        return {
            call.id: self.calls.delegated[id(call)]
            for call in self.asked
            if id(call) in self.calls.delegated
        }

    def asking(self) -> str | None:
        return next((call.id for call in self.asked if id(call) in self.calls.asking), None)

    def held_names(self) -> list[str]:
        return [
            call.name
            for call in self.asked
            if id(call) in self.calls.waiting or id(call) in self.calls.delegated
        ]

    def refuse(self, output: ToolCall, why: str) -> None:
        self.transcript.messages.append(
            Message(role="tool", tool_call_id=output.id, content=why, is_error=True)
        )
        self.steps.append(
            Step(
                tool=OUTPUT_TOOL,
                args=output.args,
                result=why,
                is_error=True,
                at=_now(),
                duration_ms=0,
            )
        )

    def accept(self, output: ToolCall) -> None:
        self.transcript.messages.append(
            Message(role="tool", tool_call_id=output.id, content="accepted")
        )
        self.steps.append(
            Step(
                tool=OUTPUT_TOOL,
                args=output.args,
                result="accepted",
                is_error=False,
                at=_now(),
                duration_ms=0,
            )
        )

    def keep_built(self) -> None:
        for call in self.asked:
            built = self.built(call)
            if built is not None and not any(built is kept for kept in self.steps):
                self.answer(call, built)


class Bridge:
    calls: _ToolCalls

    def __init__(
        self,
        run_id: str,
        agent: str,
        stop: Stop,
        on_event: EventHandler | None,
        variables: dict[str, str],
        bridging: bool,
        reachable: Tools,
        steps: list[Step],
    ) -> None:
        self.run_id = run_id
        self.agent = agent
        self.stop = stop
        self.on_event = on_event
        self.variables = variables
        self.bridging = bridging
        self.reachable = reachable
        self.steps = steps

    async def show(self, call: ToolCall, here: Foreground, stream: Pipe, text: str) -> None:
        if here.failed is not None:
            return
        try:
            await self.calls.emit(
                CommandOutput(
                    run_id=self.run_id, agent=self.agent, call_id=call.id, stream=stream, text=text
                )
            )
        except Exception as exc:
            here.failed = exc

    async def bridged(
        self,
        shell: ToolCall,
        here: Foreground,
        numbers: Iterator[int],
        name: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        if name not in self.reachable:
            return {"error": error_text(InputError(f"tools: unknown tool {name!r}"))}
        call = ToolCall(id=f"{shell.id}.{next(numbers)}", name=name, args=args)
        began = time.monotonic()
        calling = asyncio.ensure_future(self.calls.run(call, shell.id))
        while not calling.done():
            try:
                await asyncio.wait([calling])
            except asyncio.CancelledError:
                own = self.calls.foregrounds.get(id(call))
                if (
                    self.calls.runner is not None
                    and self.calls.runner.cancelling() > self.calls.calm
                ):
                    calling.cancel()
                elif requested(self.stop) == "now" and own is not None and own.live > 0:
                    own.pressed = True
                elif not any(done is call for done, _ in self.calls.ran):
                    calling.cancel()
        cut_step = self.calls.stopped(call, began, shell.id) if calling.cancelled() else None
        self.calls.foregrounds.pop(id(call), None)
        if cut_step is not None:
            self.steps.append(cut_step)
            try:
                await self.calls.hand_over(call, cut_step)
            except Exception as exc:
                here.failed = here.failed or exc
            return {"error": STOPPED}
        failed = calling.exception()
        if isinstance(failed, Exception):
            self.steps.extend(built for done, built in self.calls.ran if done is call)
            here.failed = here.failed or failed
            return {"error": error_text(failed)}
        step = calling.result()
        self.steps.append(step)
        shown = (
            step.result
            if isinstance(step.result, str)
            else [part.model_dump(mode="json") for part in step.result]
        )
        return {"error" if step.is_error else "result": shown}

    def lend(self, call: ToolCall, here: Foreground) -> None:
        if call.name != "Shell":
            return
        numbers = count(1)
        here.lent = Lent(
            env=self.variables,
            said=None
            if self.on_event is None
            else lambda stream, text: self.show(call, here, stream, text),
            call=(lambda name, args: self.bridged(call, here, numbers, name, args))
            if self.bridging
            else None,
        )

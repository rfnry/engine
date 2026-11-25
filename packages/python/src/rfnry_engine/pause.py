from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING

from rfnry_engine.batch import STOPPED, Batch, _now, _raise_first, _refusal
from rfnry_engine.conversation import _under, open_calls
from rfnry_engine.helpers import Helper, Helpers
from rfnry_engine.models import Message, Pause, Refused, ToolCall, Waiting, _call, _handed, _waited
from rfnry_engine.records import Step
from rfnry_engine.stop import Stop
from rfnry_engine.transcript import Transcript

if TYPE_CHECKING:
    from rfnry_engine.engine import _Children, _ToolCalls


class Pausing:
    def __init__(
        self,
        agent: str,
        run_id: str,
        transcript: Transcript,
        batch: Batch,
        tool_calls: _ToolCalls,
        children: _Children,
        made: list[ToolCall],
        repaired: bool,
        lead: tuple[Helper, Helpers] | None,
    ) -> None:
        self.agent = agent
        self.run_id = run_id
        self.transcript = transcript
        self.batch = batch
        self.tool_calls = tool_calls
        self.children = children
        self.made = made
        self.repaired = repaired
        self.lead = lead

    def still_open(
        self, held: Sequence[Message] = (), answers: Collection[str] = (), told: bool = False
    ) -> Pause:
        base = (
            Pause(agent=self.agent, run_id=self.run_id)
            if self.transcript.open_mark is None
            else self.transcript.open_mark
        )
        after = self.transcript.after_mark()
        answered = {message.tool_call_id for message in after if message.role == "tool"}
        pending = self.batch.pending()
        kept = {
            call_id: key
            for call_id, key in base.delegated.items()
            if call_id not in answered and call_id not in pending
        }

        def waits_of(helper: Helper) -> list[Waiting]:
            if helper.ended == "paused":
                return helper.waits
            key = self.children.key_of(helper)
            return [wait for wait in base.below if key and _under(wait.conversation, key)]

        asking = self.batch.asking()
        if asking is None and not told and base.asking not in answered:
            asking = base.asking
        return Pause(
            agent=base.agent,
            run_id=self.run_id,
            waiting=[
                *(
                    _waited(wait)
                    for wait in base.waiting
                    if wait.id not in answers and wait.call.id not in answered
                ),
                *(_waited(wait) for wait in self.batch.waiting()),
            ],
            below=[
                *(
                    _waited(wait)
                    for wait in base.below
                    if wait.id not in answers
                    and any(_under(wait.conversation, key) for key in kept.values())
                ),
                *(_waited(wait) for _, below in pending.values() for wait in below),
                *(
                    _waited(wait)
                    for helper in self.children.helpers.running_or_paused()
                    for wait in waits_of(helper)
                    if wait.id not in answers
                ),
            ],
            delegated={**kept, **{call_id: key for call_id, (key, _) in pending.items()}},
            asking=asking,
            held=list(held),
        )

    async def resume(self, mark: Pause, answers: Mapping[str, bool | Refused], stop: Stop) -> bool:
        answered = [waiting for waiting in mark.waiting if waiting.id in answers]
        taken = [waiting for waiting in mark.below if waiting.id in answers]
        told = self.lead is not None and self.lead[0].told is not None
        asking = next((call for call in self.made if told and call.id == mark.asking), None)

        def reached(key: str) -> bool:
            under = [waiting for waiting in mark.below if _under(waiting.conversation, key)]
            return not under or any(waiting.id in answers for waiting in under)

        children_again = {
            call.id: mark.delegated[call.id]
            for call in self.made
            if call.id in mark.delegated and reached(mark.delegated[call.id])
        }
        self.children.rerun.update(children_again)
        refused: list[tuple[ToolCall, Step]] = []
        for waiting in answered:
            if answers[waiting.id] is True:
                continue
            call = _call(waiting.call)
            step = Step(
                tool=call.name,
                args=call.args,
                result=_refusal(call.name, answers[waiting.id]),
                is_error=True,
                at=_now(),
                duration_ms=0,
                approved=False,
            )
            self.batch.answer(call, step)
            refused.append((call, step))
        if self.transcript.conversation is not None and (
            answered or taken or self.repaired or asking is not None
        ):
            held = [_handed(message) for message in mark.held]
            self.transcript.mark(self.still_open(held, answers, told=asking is not None))
            await self.transcript.save()
        reattached = await self.children.go_on(answers)
        failures: list[Exception] = []
        for call, step in refused:
            try:
                await self.tool_calls.answered(call, step)
            except Exception as exc:
                failures.append(exc)
        self.batch.asked = [
            *(_call(waiting.call) for waiting in answered if answers[waiting.id] is True),
            *(_call(call) for call in self.made if call.id in children_again or call is asking),
        ]
        self.tool_calls.granted.update(id(call) for call in self.batch.asked)
        try:
            if self.batch.asked:
                await self.batch.run(self.batch.asked, stop, stored=False)
        except Exception as exc:
            failures.append(exc)
        finally:
            self.tool_calls.granted.clear()
        _raise_first(failures)
        if self.transcript.conversation is not None and self.transcript.past_mark():
            after = self.still_open([_handed(message) for message in mark.held])
            self.transcript.remark(after)
            await self.transcript.save()
        return bool(answered or taken or children_again or asking or reattached)

    async def let_go(self) -> None:
        answered = self.batch.unwait(
            lambda call, step: step.model_copy(
                update={
                    "result": STOPPED,
                    "is_error": True,
                    "stopped": True,
                    "waiting": False,
                    "at": _now(),
                }
            )
        )
        for call, cut_step in answered:
            await self.tool_calls.hand_over(call, cut_step)
        given_up: list[tuple[ToolCall, Step]] = []
        still = open_calls(self.still_open())
        for call in [_call(call) for call in self.made if call.id in still]:
            held = self.children.parked.get(call.id)
            child_run_id = None if held is None else await self.children.stop_child(*held)
            cut_step = Step(
                tool=call.name,
                args=call.args,
                result=STOPPED,
                is_error=True,
                at=_now(),
                duration_ms=0,
                child_run_id=child_run_id,
                stopped=True,
            )
            self.batch.answer(call, cut_step)
            given_up.append((call, cut_step))
        for call_id, held in list(self.children.parked.items()):
            child_run_id = await self.children.stop_child(*held)
            linked = next((step for done, step in self.tool_calls.ran if done.id == call_id), None)
            if linked is not None and child_run_id is not None:
                linked.child_run_id = child_run_id
        failures: list[Exception] = []
        for call, cut_step in given_up:
            try:
                await self.tool_calls.answered(call, cut_step)
            except Exception as exc:
                failures.append(exc)
        _raise_first(failures)

    async def pause(self, heard: Sequence[Message] = ()) -> Pause:
        helpers = self.children.helpers
        await helpers.settled()
        if helpers.fatal is not None:
            raise helpers.fatal
        mark = self.still_open([*heard, *helpers.take()])
        self.transcript.mark(mark)
        return mark

    def kept_mark(self, *, top_level: bool, called_model: bool) -> Pause | None:
        return (
            self.transcript.open_mark
            if self.transcript.conversation is not None and top_level and not called_model
            else None
        )

    def end_kept(self, kept: Pause, taken: Sequence[Message]) -> None:
        if taken or self.transcript.past_mark():
            again = self.still_open([*(_handed(message) for message in kept.held), *taken])
            self.transcript.close(again)

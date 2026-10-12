import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError, DelegationDepthError, ModelError, RunawayError
from rfnry_engine.models import (
    ChoiceAnswer,
    Message,
    QuestionLike,
    Reply,
    State,
    ToolCall,
    ToolSpec,
    Usage,
)
from rfnry_engine.records import Decision, Event, Generation, Run, ToolResultReady
from rfnry_engine.stop import Stop
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import TOOLS, Answering, Desk, Scripted, delegate_triage, lookup_order, text_reply


class Kept:
    def __init__(self) -> None:
        self.runs: list[Run] = []

    async def record(self, run: Run) -> None:
        self.runs.append(run)


class Down:
    async def record(self, run: Run) -> None:
        raise OSError("storage is down")


class Breaking(Scripted):
    def __init__(self, *replies: Reply, error: Exception) -> None:
        super().__init__(*replies)
        self.error = error

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        if not self.replies:
            self.calls.append((system, list(messages), list(tools)))
            raise self.error
        return await super().generate(system, messages, tools)


class Refusing:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def decide(self, state: State, questions: Mapping[str, QuestionLike]) -> Any:
        raise self.error


def looked_up() -> Reply:
    call = ToolCall(id="c1", name="lookup_order", args={"order_id": "A-1"})
    return Reply(tool_calls=[call], stop_reason="tool_use")


def too_long() -> ModelError:
    error = ModelError("anthropic: prompt is too long")
    error.__cause__ = ValueError("413")
    return error


async def test_a_run_whose_model_call_fails_is_recorded_then_raised(fixtures: Path) -> None:
    error = too_long()
    kept = Kept()
    engine = Engine(
        fixtures, generation=Breaking(looked_up(), error=error), tools=TOOLS, refinery=kept
    )
    with pytest.raises(ModelError) as caught:
        await engine.run("support", "acme", "Where is A-1?")
    assert caught.value is error and isinstance(caught.value.__cause__, ValueError)
    [run] = kept.runs
    assert isinstance(run, Generation)
    assert run.error == "ModelError: anthropic: prompt is too long"
    assert run.reply == "" and run.stopped is None and run.stop_reason == "tool_use"
    assert [step.tool for step in run.steps] == ["lookup_order"] and run.calls == 1
    assert [m.role for m in run.conversation] == ["user", "assistant", "tool"]


async def test_a_recorder_that_fails_never_hides_why_the_run_failed(fixtures: Path) -> None:
    error = too_long()
    engine = Engine(fixtures, generation=Breaking(error=error), tools=TOOLS, refinery=Down())
    with pytest.raises(ModelError) as caught:
        await engine.run("support", "acme", "Where is A-1?")
    assert caught.value is error


async def test_a_failed_run_is_not_recorded_with_record_false(fixtures: Path) -> None:
    kept = Kept()
    engine = Engine(fixtures, generation=Breaking(error=too_long()), tools=TOOLS, refinery=kept)
    with pytest.raises(ModelError):
        await engine.run("support", "acme", "Where is A-1?", record=False)
    assert kept.runs == []


async def test_a_decision_run_whose_model_fails_is_recorded_without_answers(
    fixtures: Path,
) -> None:
    kept = Kept()
    engine = Engine(fixtures, decision=Refusing(TimeoutError("decide timed out")), refinery=kept)
    with pytest.raises(TimeoutError):
        await engine.run("triage", "acme", {"message": "It is down"})
    [run] = kept.runs
    assert isinstance(run, Decision) and run.answers == {} and run.routed_to is None
    assert run.error == "TimeoutError: decide timed out"


async def test_a_delegated_run_that_fails_stays_under_its_parent(fixtures: Path) -> None:
    kept = Kept()
    engine = Engine(
        fixtures,
        generation=Scripted(delegate_triage(), text_reply("I could not triage it.")),
        decision=Refusing(TimeoutError("decide timed out")),
        tools=TOOLS,
        refinery=kept,
    )
    run = await engine.run("support", "acme", "It is down")
    assert isinstance(run, Generation) and run.error is None
    assert run.reply == "I could not triage it."
    [child] = run.children
    assert isinstance(child, Decision) and child.error == "TimeoutError: decide timed out"
    [step] = run.steps
    assert step.child_run_id == child.run_id and step.is_error
    assert step.result == "TimeoutError: decide timed out"
    assert kept.runs == [run]


async def test_a_routed_run_that_fails_fails_its_router_and_both_are_recorded(
    fixtures: Path,
) -> None:
    kept = Kept()
    engine = Engine(
        fixtures,
        generation=Breaking(error=ModelError("anthropic: overloaded")),
        decision=Desk(),
        tools=TOOLS,
        refinery=kept,
    )
    with pytest.raises(ModelError):
        await engine.run("frontdesk", "acme", "It is down")
    [router] = kept.runs
    assert isinstance(router, Decision) and router.error == "ModelError: anthropic: overloaded"
    intent = router.answers["intent"]
    assert isinstance(intent, ChoiceAnswer) and intent.choice == "technical"
    routed = router.routed_to
    assert isinstance(routed, Generation) and routed.agent == "support"
    assert routed.error == router.error and routed.parent_run_id == router.run_id


async def test_a_routed_run_refused_before_its_first_call_leaves_routed_to_none(
    fixtures: Path,
) -> None:
    kept = Kept()
    llm = Scripted()
    engine = Engine(
        fixtures, generation=llm, decision=Desk(), tools=TOOLS, refinery=kept, max_depth=1
    )
    with pytest.raises(DelegationDepthError):
        await engine.run("frontdesk", "acme", "It is down")
    [router] = kept.runs
    assert isinstance(router, Decision) and router.routed_to is None
    assert router.error is not None and router.error.startswith("DelegationDepthError")
    assert llm.calls == []


async def test_an_approve_that_raises_fails_the_run_and_it_is_recorded(fixtures: Path) -> None:
    async def approve(agent: str, call: ToolCall) -> bool:
        raise PermissionError("approval service down")

    kept = Kept()
    engine = Engine(fixtures, generation=Scripted(looked_up()), tools=TOOLS, refinery=kept)
    with pytest.raises(PermissionError):
        await engine.run("support", "acme", "Where is A-1?", approve=approve)
    [run] = kept.runs
    assert isinstance(run, Generation)
    assert run.error == "PermissionError: approval service down"
    assert run.steps == [] and [m.role for m in run.conversation] == ["user", "assistant"]


async def test_a_run_whose_task_is_cancelled_is_not_recorded(fixtures: Path) -> None:
    waiting = asyncio.Event()

    class Slow:
        async def generate(
            self, system: str, messages: list[Message], tools: list[ToolSpec]
        ) -> Reply:
            waiting.set()
            await asyncio.sleep(60)
            return text_reply("late")

    kept = Kept()
    engine = Engine(fixtures, generation=Slow(), tools=TOOLS, refinery=kept)
    task = asyncio.ensure_future(engine.run("support", "acme", "x"))
    await waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert kept.runs == []


async def test_a_run_refused_before_its_first_model_call_is_not_recorded(
    fixtures: Path,
) -> None:
    class Verdict(BaseModel):
        refund: bool

    kept = Kept()
    llm = Scripted()
    engine = Engine(fixtures, generation=llm, tools=TOOLS, refinery=kept)
    collide = Tools([Tool("Output", "Collides", body=lookup_order)])
    with pytest.raises(ConfigurationError):
        await engine.run("support", "acme", "x", expect=Verdict, tools=collide)
    assert kept.runs == [] and llm.calls == []


async def test_a_runaway_run_is_recorded(fixtures: Path) -> None:
    kept = Kept()
    engine = Engine(
        fixtures,
        generation=Scripted(looked_up(), looked_up()),
        tools=TOOLS,
        refinery=kept,
        max_steps=2,
    )
    with pytest.raises(RunawayError):
        await engine.run("support", "acme", "Where is A-1?")
    [run] = kept.runs
    assert isinstance(run, Generation) and run.error == "RunawayError: engine: run exceeded 2 steps"
    assert len(run.steps) == 2 and run.stop_reason == "tool_use"


async def test_an_event_handler_that_raises_fails_the_run_and_it_is_recorded(
    fixtures: Path,
) -> None:
    async def closed(event: Event) -> None:
        raise ConnectionResetError("the socket closed")

    kept = Kept()
    reply = text_reply("Done.", usage=Usage(input=10, output=5), model="m-1")
    engine = Engine(fixtures, generation=Scripted(reply), tools=TOOLS, refinery=kept)
    with pytest.raises(ConnectionResetError):
        await engine.run("support", "acme", "x", on_event=closed)
    [run] = kept.runs
    assert isinstance(run, Generation) and run.error == "ConnectionResetError: the socket closed"
    assert run.calls == 1 and run.usage == Usage(input=10, output=5) and run.model == "m-1"
    assert [m.role for m in run.conversation] == ["user", "assistant"]


async def test_a_failed_run_keeps_the_step_of_a_tool_that_already_ran(fixtures: Path) -> None:
    async def closed(event: Event) -> None:
        if isinstance(event, ToolResultReady):
            raise ConnectionResetError("the socket closed")

    kept = Kept()
    engine = Engine(fixtures, generation=Scripted(looked_up()), tools=TOOLS, refinery=kept)
    with pytest.raises(ConnectionResetError):
        await engine.run("support", "acme", "Where is A-1?", on_event=closed)
    [run] = kept.runs
    assert [s.tool for s in run.steps] == ["lookup_order"]
    assert [m.role for m in run.conversation] == ["user", "assistant", "tool"]


async def test_a_failed_batch_keeps_every_step_that_ran_in_the_order_asked(
    fixtures: Path,
) -> None:
    async def slow_first(args: dict[str, Any]) -> Any:
        if args["order_id"] == "A-1":
            await asyncio.sleep(0.05)
        return await lookup_order(args)

    async def closed(event: Event) -> None:
        if isinstance(event, ToolResultReady) and event.call_id == "c1":
            raise ConnectionResetError("the socket closed")

    calls = [
        ToolCall(id="c1", name="lookup_order", args={"order_id": "A-1"}),
        ToolCall(id="c2", name="lookup_order", args={"order_id": "A-2"}),
    ]
    tools = [Tool("lookup_order", body=slow_first), *(t for t in TOOLS if t.name != "lookup_order")]
    kept = Kept()
    engine = Engine(
        fixtures,
        generation=Scripted(Reply(tool_calls=calls, stop_reason="tool_use")),
        tools=tools,
        refinery=kept,
    )
    with pytest.raises(ConnectionResetError):
        await engine.run("support", "acme", "x", on_event=closed)
    [run] = kept.runs
    assert [s.args["order_id"] for s in run.steps] == ["A-1", "A-2"]
    assert [m.tool_call_id for m in run.conversation if m.role == "tool"] == ["c1", "c2"]


async def test_a_call_cut_by_a_stop_is_recorded_once_when_its_handler_raises(
    fixtures: Path,
) -> None:
    stop = Stop()

    async def closed(event: Event) -> None:
        if isinstance(event, ToolResultReady):
            if event.step.stopped:
                raise ConnectionResetError("the socket closed")
            stop.now()
            await asyncio.sleep(1)

    kept = Kept()
    engine = Engine(fixtures, generation=Scripted(looked_up()), tools=TOOLS, refinery=kept)
    with pytest.raises(ConnectionResetError):
        await engine.run("support", "acme", "Where is A-1?", on_event=closed, stop=stop)
    [run] = kept.runs
    [step] = run.steps
    assert step.tool == "lookup_order" and step.stopped
    assert [m.role for m in run.conversation] == ["user", "assistant", "tool"]


async def test_a_failed_run_keeps_the_children_that_already_ran(fixtures: Path) -> None:
    async def closed(event: Event) -> None:
        if isinstance(event, ToolResultReady):
            raise ConnectionResetError("the socket closed")

    kept = Kept()
    engine = Engine(
        fixtures,
        generation=Scripted(delegate_triage()),
        decision=Answering(),
        tools=TOOLS,
        refinery=kept,
    )
    with pytest.raises(ConnectionResetError):
        await engine.run("support", "acme", "It is down", on_event=closed)
    [run] = kept.runs
    assert isinstance(run, Generation) and run.error == "ConnectionResetError: the socket closed"
    [child] = run.children
    assert isinstance(child, Decision) and child.agent == "triage" and child.error is None
    [step] = run.steps
    assert step.child_run_id == child.run_id

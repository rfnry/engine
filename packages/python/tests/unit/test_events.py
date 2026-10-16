import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from rfnry_engine import Event, Stop, TextDelta, ToolCallStarted, ToolResultReady
from rfnry_engine.engine import Engine
from rfnry_engine.models import Reply, ToolCall
from rfnry_engine.records import Generation
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import Scripted, Streaming, text_reply

STOPPED = "stopped by the user"


class Verdict(BaseModel):
    ok: bool


def make_tree(tmp_path: Path) -> Path:
    root = tmp_path / "agents"
    for name, front in {
        "lead": "kind: generation\ndescription: Leads\nagents: [chat]",
        "chat": "kind: generation\ndescription: Talks",
    }.items():
        (root / name).mkdir(parents=True)
        (root / name / "AGENT.md").write_text(
            f"---\nname: {name}\n{front}\n---\nYou are {name}.\n", encoding="utf-8"
        )
    return root


async def lookup(args: dict[str, Any]) -> dict[str, str]:
    return {"order": args["order_id"], "status": "shipped"}


async def broken(args: dict[str, Any]) -> str:
    raise RuntimeError("the ledger is down")


async def slow(args: dict[str, Any]) -> str:
    await asyncio.sleep(30)
    return "never"


async def refund(args: dict[str, Any]) -> str:
    raise AssertionError("a refused call must not run")


TOOLS = Tools(
    [
        Tool("lookup", body=lookup),
        Tool("broken", body=broken),
        Tool("slow", body=slow),
        Tool("refund", body=refund),
    ]
)


class Collect:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def __call__(self, event: Event) -> None:
        self.events.append(event)

    def of(self, run_id: str) -> list[ToolCallStarted | ToolResultReady]:
        return [
            e
            for e in self.events
            if isinstance(e, ToolCallStarted | ToolResultReady) and e.run_id == run_id
        ]


def check(run: Generation, calls: list[ToolCall], events: list[Any]) -> None:
    started = [e for e in events if isinstance(e, ToolCallStarted)]
    ready = [e for e in events if isinstance(e, ToolResultReady)]
    assert {e.call_id: (e.tool, e.args) for e in started} == {c.id: (c.name, c.args) for c in calls}
    assert len(ready) == len(calls) and all(e.agent == run.agent for e in events)
    steps = {e.call_id: e.step for e in ready}
    assert [steps[c.id] for c in calls] == run.steps
    for c in calls:
        assert events.index(next(e for e in started if e.call_id == c.id)) < events.index(
            next(e for e in ready if e.call_id == c.id)
        )


async def test_every_finished_call_sends_the_step_the_run_records(tmp_path: Path) -> None:
    calls = [
        ToolCall(id="c1", name="lookup", args={"order_id": "A-1"}),
        ToolCall(id="c2", name="broken", args={"why": "x"}),
        ToolCall(id="c3", name="slow", args={}),
        ToolCall(id="c4", name="refund", args={"order_id": "A-1", "amount": 30}),
        ToolCall(id="c5", name="Delegate", args={"agent": "chat", "input": "Say hi"}),
    ]
    llm = Scripted(
        Reply(tool_calls=calls, stop_reason="tool_use"),
        text_reply("Hi from chat."),
        text_reply("Done."),
    )

    async def approve(agent: str, call: ToolCall) -> bool:
        return call.name != "refund"

    collect = Collect()
    engine = Engine(make_tree(tmp_path), generation=llm, tool_timeout=0.05)
    run = await engine.run(
        "lead", "acme", "Handle A-1", tools=TOOLS, approve=approve, on_event=collect
    )
    assert isinstance(run, Generation) and run.reply == "Done."
    check(run, calls, collect.of(run.run_id))
    ok, failed, late, refused, delegated = run.steps
    assert ok.result == '{"order": "A-1", "status": "shipped"}' and not ok.is_error
    assert failed.result == "RuntimeError: the ledger is down" and failed.is_error
    assert late.result == "TimeoutError: slow timed out after 0.05s" and late.is_error
    assert refused.result == "NotApproved: refund was not approved" and refused.approved is False
    [child] = run.children
    assert delegated.child_run_id == child.run_id and delegated.result == "Hi from chat."
    assert [s.approved for s in (ok, failed, late, delegated)] == [True] * 4


async def test_a_call_cut_off_by_stop_sends_its_stopped_step(tmp_path: Path) -> None:
    calls = [
        ToolCall(id="c1", name="slow", args={"order_id": "A-1"}),
        ToolCall(id="c2", name="refund", args={"order_id": "A-2"}),
    ]
    started = asyncio.Event()

    async def tracked(args: dict[str, Any]) -> str:
        started.set()
        return await slow(args)

    asked = asyncio.Event()

    async def approve(agent: str, call: ToolCall) -> bool:
        if call.name == "refund":
            asked.set()
            await asyncio.sleep(30)
        return True

    collect = Collect()
    stop = Stop()
    tools = Tools([Tool("slow", body=tracked), Tool("refund", body=refund)])
    engine = Engine(
        make_tree(tmp_path), generation=Scripted(Reply(tool_calls=calls, stop_reason="tool_use"))
    )
    running = asyncio.create_task(
        engine.run(
            "lead", "acme", "Refund", tools=tools, approve=approve, stop=stop, on_event=collect
        )
    )
    await started.wait()
    await asked.wait()
    stop.now()
    run = await asyncio.wait_for(running, 2)
    assert isinstance(run, Generation) and run.stopped == "now"
    check(run, calls, collect.of(run.run_id))
    assert [(s.result, s.is_error, s.stopped) for s in run.steps] == [(STOPPED, True, True)] * 2


class StopOn(Collect):
    def __init__(self, stop: Stop, kind: type[BaseModel], pause: float) -> None:
        super().__init__()
        self.stop, self.kind, self.pause = stop, kind, pause
        self.handled: list[Event] = []

    async def __call__(self, event: Event) -> None:
        await super().__call__(event)
        if isinstance(event, self.kind) and not self.handled:
            self.stop.now()
            await asyncio.sleep(self.pause)
            self.handled.append(event)


@pytest.mark.parametrize("pause", [0, 0.01])
async def test_a_stop_from_a_result_handler_sends_each_result_once(
    tmp_path: Path, pause: float
) -> None:
    calls = [
        ToolCall(id="c1", name="lookup", args={"order_id": "A-1"}),
        ToolCall(id="c2", name="slow", args={}),
    ]
    stop = Stop()
    handler = StopOn(stop, ToolResultReady, pause)
    llm = Scripted(Reply(tool_calls=calls, stop_reason="tool_use"))
    run = await asyncio.wait_for(
        Engine(make_tree(tmp_path), generation=llm).run(
            "lead", "acme", "Check A-1", tools=TOOLS, stop=stop, on_event=handler
        ),
        2,
    )
    assert isinstance(run, Generation) and run.stopped == "now"
    events = handler.of(run.run_id)
    check(run, calls, events)
    assert [(s.result, s.stopped) for s in run.steps] == [
        ('{"order": "A-1", "status": "shipped"}', False),
        (STOPPED, True),
    ]
    first = [e for e in events if isinstance(e, ToolResultReady) and e.call_id == "c1"]
    assert handler.handled == first


async def test_a_stop_from_a_result_handler_answers_a_pending_output_once(
    tmp_path: Path,
) -> None:
    calls = [
        ToolCall(id="c1", name="lookup", args={"order_id": "A-1"}),
        ToolCall(id="o1", name="Output", args={"ok": True}),
    ]
    stop = Stop()
    handler = StopOn(stop, ToolResultReady, 0.01)
    llm = Scripted(Reply(tool_calls=calls, stop_reason="tool_use"))
    run = await asyncio.wait_for(
        Engine(make_tree(tmp_path), generation=llm).run(
            "lead", "acme", "Check A-1", tools=TOOLS, expect=Verdict, stop=stop, on_event=handler
        ),
        2,
    )
    assert isinstance(run, Generation) and run.stopped == "now"
    ready = [e for e in handler.of(run.run_id) if isinstance(e, ToolResultReady)]
    assert [e.call_id for e in ready] == ["c1", "o1"]
    assert [e.step for e in ready] == run.steps
    assert [(s.tool, s.stopped) for s in run.steps] == [("lookup", False), ("Output", True)]


async def test_a_stop_from_a_start_handler_sends_each_event_once(tmp_path: Path) -> None:
    calls = [
        ToolCall(id="c1", name="lookup", args={"order_id": "A-1"}),
        ToolCall(id="c2", name="slow", args={}),
    ]
    stop = Stop()
    handler = StopOn(stop, ToolCallStarted, 0.01)
    llm = Scripted(Reply(tool_calls=calls, stop_reason="tool_use"))
    run = await asyncio.wait_for(
        Engine(make_tree(tmp_path), generation=llm).run(
            "lead", "acme", "Check A-1", tools=TOOLS, stop=stop, on_event=handler
        ),
        2,
    )
    assert isinstance(run, Generation) and run.stopped == "now"
    check(run, calls, handler.of(run.run_id))
    assert [(s.result, s.stopped) for s in run.steps] == [(STOPPED, True)] * 2


async def test_a_stop_from_a_text_handler_sends_each_delta_once(tmp_path: Path) -> None:
    looked = ToolCall(id="c1", name="lookup", args={"order_id": "A-1"})
    stop = Stop()
    handler = StopOn(stop, TextDelta, 0.01)
    llm = Streaming(Reply(text="Let me look.", tool_calls=[looked], stop_reason="tool_use"))
    run = await asyncio.wait_for(
        Engine(make_tree(tmp_path), generation=llm).run(
            "lead", "acme", "Check A-1", tools=TOOLS, stop=stop, on_event=handler
        ),
        2,
    )
    assert isinstance(run, Generation) and run.stopped == "now" and run.reply == "Let "
    assert [e.text for e in handler.events if isinstance(e, TextDelta)] == ["Let "]
    assert handler.of(run.run_id) == [] and run.steps == []


async def test_cancelling_a_run_a_stop_holds_on_a_result_handler_cancels_the_handler(
    tmp_path: Path,
) -> None:
    stop, entered, left = Stop(), asyncio.Event(), asyncio.Event()

    async def stuck(event: Event) -> None:
        if isinstance(event, ToolResultReady):
            stop.now()
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                left.set()

    before = asyncio.all_tasks()
    looked = ToolCall(id="c1", name="lookup", args={"order_id": "A-1"})
    llm = Scripted(Reply(tool_calls=[looked], stop_reason="tool_use"))
    running = asyncio.create_task(
        Engine(make_tree(tmp_path), generation=llm).run(
            "lead", "acme", "Check A-1", tools=TOOLS, stop=stop, on_event=stuck
        )
    )
    await entered.wait()
    await asyncio.sleep(0.01)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    await asyncio.wait_for(left.wait(), 1)
    assert asyncio.all_tasks() <= before


async def test_a_childs_text_carries_its_own_run_and_its_own_turns(tmp_path: Path) -> None:
    delegate = ToolCall(id="d", name="Delegate", args={"agent": "chat", "input": "Say hi"})
    llm = Scripted(
        Reply(text="Asking chat.", tool_calls=[delegate], stop_reason="tool_use"),
        text_reply("Hi from chat."),
        text_reply("Chat says hi."),
    )
    collect = Collect()
    run = await Engine(make_tree(tmp_path), generation=llm).run(
        "lead", "acme", "Greet", on_event=collect
    )
    assert isinstance(run, Generation) and run.reply == "Chat says hi."
    [child] = run.children
    said = [(e.run_id, e.agent, e.turn, e.text) for e in collect.events if isinstance(e, TextDelta)]
    assert said == [
        (run.run_id, "lead", 0, "Asking chat."),
        (child.run_id, "chat", 0, "Hi from chat."),
        (run.run_id, "lead", 1, "Chat says hi."),
    ]

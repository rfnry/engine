import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError
from rfnry_engine.models import Answer, ChoiceAnswer, Reply, ToolCall
from rfnry_engine.records import Decision, Generation
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.refinery.storage import Storage
from rfnry_engine.stop import Stop
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import Fixed, MemoryBucket, Scripted, text_reply

AGENTS = {
    "desk": "kind: generation\ndescription: Helps customers",
    "lead": "kind: generation\ndescription: Leads\nagents: [desk]",
    "router": "kind: decision\ndescription: Sorts\nroute:\n  by: intent\n  to: {help: desk}",
}
INTENT = (
    "---\ntype: choice\noptions:\n  help: Needs help\n  other: Anything else\n---\nWhat is this?\n"
)


def make_tree(tmp_path: Path) -> Path:
    root = tmp_path / "agents"
    for name, front in AGENTS.items():
        (root / name).mkdir(parents=True)
        (root / name / "AGENT.md").write_text(
            f"---\n{front}\n---\nYou are {name}.\n", encoding="utf-8"
        )
        if "kind: decision" in front:
            (root / name / "questions").mkdir()
            (root / name / "questions" / "intent.md").write_text(INTENT, encoding="utf-8")
    return root


def use(name: str, **args: Any) -> Reply:
    return Reply(
        tool_calls=[ToolCall(id=f"c-{name}", name=name, args=args)], stop_reason="tool_use"
    )


class Refunds:
    def __init__(self) -> None:
        self.done: list[dict[str, Any]] = []

    def tools(self) -> Tools:
        async def refund(args: dict[str, Any]) -> str:
            self.done.append(args)
            return "refunded"

        return Tools([Tool("refund", "Refund an order", body=refund)])


class Approver:
    def __init__(self, allow: bool = True, delay: float = 0.0) -> None:
        self.allow, self.delay = allow, delay
        self.asked: list[tuple[str, str, dict[str, Any]]] = []

    async def __call__(self, agent: str, call: ToolCall) -> bool:
        self.asked.append((agent, call.name, call.args))
        await asyncio.sleep(self.delay)
        return self.allow


async def test_without_approve_every_call_runs_and_nothing_is_recorded_as_asked(
    tmp_path: Path,
) -> None:
    refunds = Refunds()
    engine = Engine(
        make_tree(tmp_path), generation=Scripted(use("refund", order="A-1"), text_reply("done"))
    )
    run = await engine.run("desk", "acme", "refund A-1", tools=refunds.tools())
    assert isinstance(run, Generation) and refunds.done == [{"order": "A-1"}]
    assert run.steps[0].approved is None


async def test_an_approved_call_runs_and_the_step_says_so(tmp_path: Path) -> None:
    refunds, approver = Refunds(), Approver()
    engine = Engine(
        make_tree(tmp_path), generation=Scripted(use("refund", order="A-1"), text_reply("done"))
    )
    run = await engine.run("desk", "acme", "refund A-1", tools=refunds.tools(), approve=approver)
    assert isinstance(run, Generation) and approver.asked == [("desk", "refund", {"order": "A-1"})]
    assert (
        refunds.done == [{"order": "A-1"}]
        and run.steps[0].approved is True
        and not run.steps[0].is_error
    )


async def test_a_refused_call_never_runs_and_the_model_reads_why(tmp_path: Path) -> None:
    refunds = Refunds()
    llm = Scripted(use("refund", order="A-1"), text_reply("I cannot refund that."))
    engine = Engine(make_tree(tmp_path), generation=llm)
    run = await engine.run(
        "desk", "acme", "refund A-1", tools=refunds.tools(), approve=Approver(allow=False)
    )
    assert (
        isinstance(run, Generation) and refunds.done == [] and run.reply == "I cannot refund that."
    )
    step = run.steps[0]
    assert (step.approved, step.is_error, step.result) == (
        False,
        True,
        "NotApproved: refund was not approved",
    )
    assert llm.calls[1][1][-1].content == "NotApproved: refund was not approved"


async def test_waiting_for_approval_does_not_count_against_the_tool_timeout(tmp_path: Path) -> None:
    refunds = Refunds()
    engine = Engine(
        make_tree(tmp_path),
        generation=Scripted(use("refund", order="A-1"), text_reply("done")),
        tool_timeout=0.05,
    )
    run = await engine.run("desk", "acme", "x", tools=refunds.tools(), approve=Approver(delay=0.3))
    assert isinstance(run, Generation)
    assert refunds.done == [{"order": "A-1"}] and not run.steps[0].is_error


async def test_stopping_now_ends_a_wait_for_approval(tmp_path: Path) -> None:
    refunds, stop = Refunds(), Stop()
    engine = Engine(
        make_tree(tmp_path), generation=Scripted(use("refund", order="A-1"), text_reply("never"))
    )
    running = asyncio.create_task(
        engine.run(
            "desk", "acme", "x", tools=refunds.tools(), approve=Approver(delay=30), stop=stop
        )
    )
    await asyncio.sleep(0.1)
    stop.now()
    run = await asyncio.wait_for(running, 5)
    assert isinstance(run, Generation) and run.stopped == "now" and refunds.done == []
    assert run.steps[0].stopped and run.steps[0].approved is None


async def test_stopping_now_ends_a_wait_for_approval_of_a_delegate(tmp_path: Path) -> None:
    stop, asked, never = Stop(), asyncio.Event(), asyncio.Event()

    async def approve(agent: str, call: ToolCall) -> bool:
        asked.set()
        await never.wait()
        return True

    llm = Scripted(use("Delegate", agent="desk", input="x"), text_reply("never"))
    engine = Engine(make_tree(tmp_path), generation=llm)
    running = asyncio.create_task(engine.run("lead", "acme", "x", approve=approve, stop=stop))
    await asyncio.wait_for(asked.wait(), 5)
    stop.now()
    run = await asyncio.wait_for(running, 5)
    assert isinstance(run, Generation) and run.stopped == "now" and run.children == []
    [step] = run.steps
    assert step.tool == "Delegate" and step.stopped and step.approved is None


@pytest.mark.parametrize("tool", ["refund", "Delegate"])
async def test_each_press_cancels_a_wait_for_approval_once(tmp_path: Path, tool: str) -> None:
    refunds, stop = Refunds(), Stop()
    asked, release = asyncio.Event(), asyncio.Event()
    cancels, cleaned = 0, False

    async def approve(agent: str, call: ToolCall) -> bool:
        nonlocal cancels, cleaned
        asked.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancels += 1
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancels += 1
            cleaned = True
            raise
        return True

    call = (
        use("refund", order="A-1") if tool == "refund" else use("Delegate", agent="desk", input="x")
    )
    engine = Engine(make_tree(tmp_path), generation=Scripted(call, text_reply("never")))
    agent = "desk" if tool == "refund" else "lead"
    running = asyncio.create_task(
        engine.run(agent, "acme", "x", tools=refunds.tools(), approve=approve, stop=stop)
    )
    try:
        await asyncio.wait_for(asked.wait(), 5)
        stop.now()
        await asyncio.sleep(0.05)
        assert cancels == 1 and not running.done()
        stop.now()
        await asyncio.sleep(0.05)
        assert cancels == 2 and not running.done()
    finally:
        release.set()
    run = await asyncio.wait_for(running, 5)
    assert isinstance(run, Generation) and run.stopped == "now" and cleaned
    [step] = run.steps
    assert step.tool == tool and step.stopped and step.approved is None


async def test_delegated_children_are_asked_too(tmp_path: Path) -> None:
    refunds, approver = Refunds(), Approver()
    llm = Scripted(
        use("Delegate", agent="desk", input="refund A-1"),
        use("refund", order="A-1"),
        text_reply("child done"),
        text_reply("lead done"),
    )
    engine = Engine(make_tree(tmp_path), generation=llm)
    run = await engine.run("lead", "acme", "x", tools=refunds.tools(), approve=approver)
    assert isinstance(run, Generation) and run.reply == "lead done"
    assert [(agent, name) for agent, name, _ in approver.asked] == [
        ("lead", "Delegate"),
        ("desk", "refund"),
    ]
    child = run.children[0]
    assert isinstance(child, Generation) and child.steps[0].approved is True


async def test_a_routed_run_is_asked_too(tmp_path: Path) -> None:
    refunds, approver = Refunds(), Approver()
    help_: dict[str, Answer] = {
        "intent": ChoiceAnswer(choice="help", probabilities={"help": 0.9, "other": 0.1})
    }
    engine = Engine(
        make_tree(tmp_path),
        generation=Scripted(use("refund", order="A-1"), text_reply("done")),
        decision=Fixed(help_),
    )
    run = await engine.run("router", "acme", "refund A-1", tools=refunds.tools(), approve=approver)
    assert isinstance(run, Decision) and isinstance(run.routed_to, Generation)
    assert approver.asked == [("desk", "refund", {"order": "A-1"})]


async def test_an_agent_as_a_tool_and_engine_tools_carry_approve(tmp_path: Path) -> None:
    refunds, approver = Refunds(), Approver(allow=False)
    llm = Scripted(
        use("refund", order="A-1"),
        text_reply("refused"),
        use("refund", order="A-2"),
        text_reply("refused"),
    )
    engine = Engine(make_tree(tmp_path), generation=llm)
    consult = engine.tool("desk", scope="acme", tools=refunds.tools(), approve=approver)
    assert await consult.dispatch("desk", {"input": "refund A-1"}) == "refused"
    surface = EngineTools(
        engine, agent="lead", scope="acme", tools=refunds.tools(), approve=approver
    )
    assert await surface.dispatch("Delegate", {"agent": "desk", "input": "refund A-2"}) == "refused"
    assert refunds.done == [] and [args for _, _, args in approver.asked] == [
        {"order": "A-1"},
        {"order": "A-2"},
    ]


async def test_an_approve_that_answers_something_else_is_the_consumers_bug(tmp_path: Path) -> None:
    async def maybe(agent: str, call: ToolCall) -> Any:
        return "yes"

    engine = Engine(
        make_tree(tmp_path), generation=Scripted(use("refund", order="A-1"), text_reply("x"))
    )
    with pytest.raises(
        ConfigurationError, match=r"^engine: approve must return True or False; got 'yes'$"
    ):
        await engine.run("desk", "acme", "x", tools=Refunds().tools(), approve=maybe)


async def test_the_exported_generation_row_keeps_each_approval(
    tmp_path: Path, storage: Storage
) -> None:
    refinery = Refinery(storage)
    llm = Scripted(use("refund", order="A-1"), text_reply("cannot"))
    engine = Engine(make_tree(tmp_path), generation=llm, refinery=refinery)
    await engine.run("desk", "acme", "x", tools=Refunds().tools(), approve=Approver(allow=False))
    bucket = MemoryBucket()
    await refinery.export("acme", bucket)
    key = next(k for k in bucket.objects if "/generation/" in k)
    row = json.loads(bucket.objects[key].decode().splitlines()[0])
    assert [(s["tool"], s["approved"]) for s in row["steps"]] == [("refund", False)]

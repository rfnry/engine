import asyncio
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine import Stop
from rfnry_engine.engine import Engine
from rfnry_engine.models import Message, QuestionLike, Reply, ToolCall, ToolSpec
from rfnry_engine.records import Decision, Event, Generation, Run
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import answer_to, text_reply

STOPPED = "stopped by the user"


def tree(root: Path, **agents: str) -> Path:
    for name, front in agents.items():
        (root / name).mkdir(parents=True)
        (root / name / "AGENT.md").write_text(
            f"---\nname: {name}\nkind: generation\nmodel: strong\ndescription: {name.title()}\n{front}---\nYou are {name}.\n",
            encoding="utf-8",
        )
    return root


def use(*calls: ToolCall) -> Reply:
    return Reply(text="", tool_calls=list(calls), stop_reason="tool_use")


def consulting(call_id: str = "c1") -> Reply:
    return use(ToolCall(id=call_id, name="billing", args={"input": "Was I charged twice?"}))


class Kept:
    def __init__(self) -> None:
        self.runs: list[Run] = []

    async def record(self, run: Run) -> None:
        self.runs.append(run)

    def of(self, agent: str) -> list[Generation]:
        return [run for run in self.runs if isinstance(run, Generation) and run.agent == agent]


class Script:
    def __init__(self, *replies: Reply | tuple[float, Reply]) -> None:
        self.replies = [item if isinstance(item, tuple) else (0.0, item) for item in replies]
        self.calls = 0

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        self.calls += 1
        if not self.replies:
            raise AssertionError("Script ran out of replies")
        delay, reply = self.replies.pop(0)
        await asyncio.sleep(delay)
        return reply


class Gated:
    def __init__(self) -> None:
        self.gate = asyncio.Event()
        self.calls = 0

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        self.calls += 1
        await self.gate.wait()
        return text_reply("Refunded.")


async def until(check: Callable[[], bool], what: str) -> None:
    try:
        async with asyncio.timeout(5):
            while not check():
                await asyncio.sleep(0.005)
    except TimeoutError:
        pytest.fail(f"{what} never happened")


async def ping(args: dict[str, Any]) -> str:
    return "pong"


PING = Tools([Tool("ping", "Ping", body=ping)])


def engines(tmp_path: Path, outer: Any, inner: Any, kept: Kept) -> tuple[Engine, Engine]:
    front = Engine(tree(tmp_path / "front", support=""), generation=outer, refinery=kept)
    back = Engine(tree(tmp_path / "back", billing=""), generation=inner, refinery=kept)
    return front, back


async def test_stop_now_reaches_a_consult_and_keeps_its_record(tmp_path: Path) -> None:
    kept = Kept()
    billing = Gated()
    front, back = engines(tmp_path, Script(consulting(), text_reply("never")), billing, kept)
    stop = Stop()
    task = asyncio.create_task(
        front.run("support", "acme", "x", tools=back.tool("billing", scope="acme"), stop=stop)
    )
    await until(lambda: billing.calls == 1, "the consult's model call")
    stop.now()
    run = await asyncio.wait_for(task, 2)
    assert isinstance(run, Generation) and run.stopped == "now"
    step = run.steps[0]
    assert (step.result, step.is_error, step.stopped) == (STOPPED, True, True)
    [consult] = kept.of("billing")
    assert consult.stopped == "now" and consult.parent_run_id is None


async def test_after_step_reaches_a_consult(tmp_path: Path) -> None:
    kept = Kept()
    billing = Script((0.2, use(ToolCall(id="p1", name="ping", args={}))), text_reply("never"))
    front, back = engines(tmp_path, Script(consulting(), text_reply("never")), billing, kept)
    stop = Stop()
    consult = back.tool("billing", scope="acme", tools=PING)
    task = asyncio.create_task(front.run("support", "acme", "x", tools=consult, stop=stop))
    await until(lambda: billing.calls == 1, "the consult's model call")
    stop.after_step()
    run = await asyncio.wait_for(task, 2)
    assert isinstance(run, Generation) and run.stopped == "after_step"
    assert (run.steps[0].result, run.steps[0].stopped) == (STOPPED, True)
    [inner] = kept.of("billing")
    assert inner.stopped == "after_step" and billing.calls == 1
    assert [step.result for step in inner.steps] == ["pong"]


async def test_the_caller_s_handler_sees_the_consult_s_events(tmp_path: Path) -> None:
    kept = Kept()
    billing = Script(use(ToolCall(id="p1", name="ping", args={})), text_reply("Refunded."))
    front, back = engines(tmp_path, Script(consulting(), text_reply("Done.")), billing, kept)
    seen: list[tuple[str, str]] = []

    async def show(event: Event) -> None:
        seen.append((event.agent, type(event).__name__))

    consult = back.tool("billing", scope="acme", tools=PING)
    run = await front.run("support", "acme", "x", tools=consult, on_event=show)
    assert isinstance(run, Generation) and run.reply == "Done."
    assert ("billing", "ToolCallStarted") in seen and ("billing", "ToolResultReady") in seen
    assert ("support", "ToolCallStarted") in seen
    assert run.steps[0].result == "Refunded." and not run.steps[0].stopped


async def test_concurrent_runs_keep_their_own_stops(tmp_path: Path) -> None:
    kept = Kept()
    billing = Gated()
    outer = Script(consulting(), consulting(), text_reply("B done."))
    front, back = engines(tmp_path, outer, billing, kept)
    consult = back.tool("billing", scope="acme")
    first, second = Stop(), Stop()
    a = asyncio.create_task(front.run("support", "acme", "a", tools=consult, stop=first))
    await until(lambda: billing.calls == 1, "the first consult")
    b = asyncio.create_task(front.run("support", "acme", "b", tools=consult, stop=second))
    await until(lambda: billing.calls == 2, "the second consult")
    first.now()
    stopped = await asyncio.wait_for(a, 2)
    assert isinstance(stopped, Generation) and stopped.stopped == "now"
    assert not b.done()
    billing.gate.set()
    finished = await asyncio.wait_for(b, 2)
    assert isinstance(finished, Generation) and finished.stopped is None
    assert finished.reply == "B done." and finished.steps[0].result == "Refunded."
    assert sorted(str(run.stopped) for run in kept.of("billing")) == ["None", "now"]


async def test_outside_a_run_the_getter_gives_each_call_its_turn_s_stop(tmp_path: Path) -> None:
    kept = Kept()
    billing = Gated()
    back = Engine(tree(tmp_path / "back", billing=""), generation=billing, refinery=kept)
    seen: list[str] = []

    async def show(event: Event) -> None:
        seen.append(event.agent)

    turn = {"stop": Stop()}
    consult = back.tool("billing", scope="acme", on_event=show, stop=lambda: turn["stop"])
    first = asyncio.create_task(consult.dispatch("billing", {"input": "Was I charged twice?"}))
    await until(lambda: billing.calls == 1, "the first consult")
    turn["stop"].now()
    await asyncio.wait_for(first, 2)
    turn["stop"] = Stop()
    billing.gate.set()
    assert await consult.dispatch("billing", {"input": "And now?"}) == "Refunded."
    assert [run.stopped for run in kept.of("billing")] == ["now", None]
    assert set(seen) == {"billing"}


async def test_explicit_hooks_win_over_the_caller_s(tmp_path: Path) -> None:
    kept = Kept()
    billing = Script(use(ToolCall(id="p1", name="ping", args={})), (30.0, text_reply("never")))
    outer = Script(consulting(), text_reply("Carried on."))
    front, back = engines(tmp_path, outer, billing, kept)
    outer_seen: list[str] = []
    own_seen: list[str] = []

    async def outer_show(event: Event) -> None:
        outer_seen.append(event.agent)

    async def own_show(event: Event) -> None:
        own_seen.append(event.agent)

    own = Stop()
    consult = back.tool("billing", scope="acme", tools=PING, on_event=own_show, stop=lambda: own)
    task = asyncio.create_task(
        front.run("support", "acme", "x", tools=consult, on_event=outer_show, stop=Stop())
    )
    await until(lambda: billing.calls == 2, "the consult's second model call")
    own.now()
    run = await asyncio.wait_for(task, 2)
    assert isinstance(run, Generation) and run.stopped is None and run.reply == "Carried on."
    assert (run.steps[0].result, run.steps[0].stopped) == (STOPPED, True)
    assert "billing" not in outer_seen and set(own_seen) == {"billing"}
    [inner] = kept.of("billing")
    assert inner.stopped == "now"


async def test_engine_tools_delegate_takes_the_hooks_outside_a_run(tmp_path: Path) -> None:
    root = tree(tmp_path / "tree", lead="agents: [chat]\n", chat="")
    kept = Kept()
    chat = Gated()
    engine = Engine(root, generation=chat, refinery=kept)
    seen: list[str] = []

    async def show(event: Event) -> None:
        seen.append(event.agent)

    turn = {"stop": Stop()}
    tools = EngineTools(
        engine, agent="lead", scope="acme", on_event=show, stop=lambda: turn["stop"]
    )
    task = asyncio.create_task(tools.dispatch("Delegate", {"agent": "chat", "input": "hi"}))
    await until(lambda: chat.calls == 1, "the delegated run")
    turn["stop"].now()
    await asyncio.wait_for(task, 2)
    [run] = kept.of("chat")
    assert run.stopped == "now"
    turn["stop"] = Stop()
    chat.gate.set()
    assert await tools.dispatch("Delegate", {"agent": "chat", "input": "again"}) == "Refunded."
    assert set(seen) == {"chat"}


class Deciding:
    def __init__(self, seconds: float = 0.0) -> None:
        self.seconds = seconds
        self.calls = 0

    async def decide(self, state: Any, questions: Mapping[str, QuestionLike]) -> dict[str, Any]:
        self.calls += 1
        await asyncio.sleep(self.seconds)
        return {qid: answer_to(question) for qid, question in questions.items()}


def decision(root: Path, name: str, front: str = "") -> None:
    (root / name / "questions").mkdir(parents=True)
    (root / name / "AGENT.md").write_text(
        f"---\nname: {name}\nkind: decision\nmodel: judge\ndescription: Sorts\n{front}---\nSort it.\n",
        encoding="utf-8",
    )
    (root / name / "questions" / "intent.md").write_text(
        "---\ntype: choice\noptions:\n  other: Anything else\n  billing: Payments\n---\nWhich?\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("through", ["engine.tool", "Delegate"])
async def test_a_stop_cuts_a_decision_run_at_once(tmp_path: Path, through: str) -> None:
    kept = Kept()
    deciding = Deciding(5.0)
    if through == "engine.tool":
        decision(tmp_path / "back", "triage")
        back = Engine(tmp_path / "back", decision=deciding, refinery=kept)
        asking = use(ToolCall(id="c1", name="triage", args={"input": "x"}))
        extra: dict[str, Any] = {"tools": back.tool("triage", scope="acme")}
        root = tree(tmp_path / "front", support="")
    else:
        root = tree(tmp_path / "front", support="agents: [triage]\n")
        decision(root, "triage")
        asking = use(ToolCall(id="d1", name="Delegate", args={"agent": "triage", "input": "x"}))
        extra = {}
    front = Engine(
        root, generation=Script(asking, text_reply("never")), decision=deciding, refinery=kept
    )
    stop = Stop()
    task = asyncio.create_task(front.run("support", "acme", "x", stop=stop, **extra))
    await until(lambda: deciding.calls == 1, "the decision call")
    began = time.monotonic()
    stop.now()
    run = await asyncio.wait_for(task, 2)
    assert time.monotonic() - began < 0.5
    assert isinstance(run, Generation) and run.stopped == "now"
    assert (run.steps[0].result, run.steps[0].stopped) == (STOPPED, True)
    assert [r.agent for r in kept.runs] == ["support"]


async def test_consults_a_tool_starts_in_tasks_of_their_own_keep_their_records(
    tmp_path: Path,
) -> None:
    kept = Kept()
    billing = Gated()
    back = Engine(tree(tmp_path / "back", billing=""), generation=billing, refinery=kept)
    consult = back.tool("billing", scope="acme")

    async def both(args: dict[str, Any]) -> str:
        async with asyncio.TaskGroup() as group:
            first = group.create_task(consult.dispatch("billing", {"input": "a"}))
            second = group.create_task(consult.dispatch("billing", {"input": "b"}))
        return f"{first.result()} {second.result()}"

    outer = Script(use(ToolCall(id="b1", name="both", args={})), text_reply("never"))
    front = Engine(tree(tmp_path / "front", support=""), generation=outer, refinery=kept)
    stop = Stop()
    fan = Tools([Tool("both", "Ask billing twice", body=both)])
    task = asyncio.create_task(front.run("support", "acme", "x", tools=fan, stop=stop))
    await until(lambda: billing.calls == 2, "both consults")
    stop.now()
    run = await asyncio.wait_for(task, 2)
    assert isinstance(run, Generation) and run.stopped == "now"
    assert (run.steps[0].result, run.steps[0].stopped) == (STOPPED, True)
    assert [r.stopped for r in kept.of("billing")] == ["now", "now"]


class Writing(Kept):
    def __init__(self) -> None:
        super().__init__()
        self.writing = asyncio.Event()

    async def record(self, run: Run) -> None:
        if run.agent == "billing":
            self.writing.set()
            await asyncio.sleep(0.3)
        await super().record(run)


@pytest.mark.parametrize("further", [False, True])
async def test_a_consult_s_record_is_written_through_every_stop(
    tmp_path: Path, further: bool
) -> None:
    kept = Writing()
    billing = Gated()
    front = Engine(
        tree(tmp_path / "front", support=""),
        generation=Script(consulting(), text_reply("never")),
        refinery=kept,
    )
    back = Engine(tree(tmp_path / "back", billing=""), generation=billing, refinery=kept)
    stop = Stop()
    consult = back.tool("billing", scope="acme")
    task = asyncio.create_task(front.run("support", "acme", "x", tools=consult, stop=stop))
    await until(lambda: billing.calls == 1, "the consult")
    stop.now()
    await asyncio.wait_for(kept.writing.wait(), 2)
    if further:
        stop.now()
    run = await asyncio.wait_for(task, 2)
    assert isinstance(run, Generation) and run.stopped == "now"
    assert [r.stopped for r in kept.of("billing")] == ["now"]


async def test_a_consult_whose_routed_run_was_stopped_reads_stopped(tmp_path: Path) -> None:
    kept = Kept()
    billing = Gated()
    back_root = tree(tmp_path / "back", billing="")
    decision(back_root, "desk", "route:\n  by: intent\n  to: {billing: billing}\n")
    back = Engine(back_root, generation=billing, decision=Deciding(), refinery=kept)
    outer = Script(use(ToolCall(id="c1", name="desk", args={"input": "x"})), text_reply("never"))
    front = Engine(tree(tmp_path / "front", support=""), generation=outer, refinery=kept)
    stop = Stop()
    consult = back.tool("desk", scope="acme")
    task = asyncio.create_task(front.run("support", "acme", "x", tools=consult, stop=stop))
    await until(lambda: billing.calls == 1, "the routed run")
    stop.now()
    run = await asyncio.wait_for(task, 2)
    assert isinstance(run, Generation)
    assert (run.steps[0].result, run.steps[0].stopped) == (STOPPED, True)
    [desk] = [r for r in kept.runs if isinstance(r, Decision)]
    assert isinstance(desk.routed_to, Generation) and desk.routed_to.stopped == "now"


async def test_a_task_that_outlives_its_call_does_not_take_the_call_s_hooks(
    tmp_path: Path,
) -> None:
    kept = Kept()
    billing = Gated()
    back = Engine(tree(tmp_path / "back", billing=""), generation=billing, refinery=kept)
    consult = back.tool("billing", scope="acme")
    go = asyncio.Event()
    later: list[asyncio.Task[Any]] = []

    async def spawn(args: dict[str, Any]) -> str:
        async def after() -> Any:
            await go.wait()
            return await consult.dispatch("billing", {"input": "later"})

        later.append(asyncio.create_task(after()))
        return "spawned"

    seen: list[str] = []

    async def show(event: Event) -> None:
        seen.append(event.agent)

    outer = Script(use(ToolCall(id="s1", name="spawn", args={})), text_reply("Done."))
    front = Engine(tree(tmp_path / "front", support=""), generation=outer, refinery=kept)
    stop = Stop()
    spawning = Tools([Tool("spawn", "Spawn", body=spawn)])
    run = await front.run("support", "acme", "x", tools=spawning, on_event=show, stop=stop)
    assert isinstance(run, Generation) and run.reply == "Done."
    stop.now()
    go.set()
    billing.gate.set()
    assert await asyncio.wait_for(later[0], 2) == "Refunded."
    assert "billing" not in seen
    [late] = kept.of("billing")
    assert late.stopped is None


async def test_a_getter_returning_none_leaves_the_consult_outside_the_caller_s_stop(
    tmp_path: Path,
) -> None:
    kept = Kept()
    billing = Gated()
    front, back = engines(tmp_path, Script(consulting(), text_reply("never")), billing, kept)
    stop = Stop()
    consult = back.tool("billing", scope="acme", stop=lambda: None)
    task = asyncio.create_task(front.run("support", "acme", "x", tools=consult, stop=stop))
    await until(lambda: billing.calls == 1, "the consult")
    stop.now()
    run = await asyncio.wait_for(task, 2)
    assert isinstance(run, Generation)
    assert (run.steps[0].result, run.steps[0].stopped) == (STOPPED, True)
    assert kept.of("billing") == []


@pytest.mark.parametrize("order", ["after", "beside"])
async def test_the_rest_of_a_call_is_cut_once_its_consult_took_the_stop(
    tmp_path: Path, order: str
) -> None:
    kept = Kept()
    billing = Gated()
    back = Engine(tree(tmp_path / "back", billing=""), generation=billing, refinery=kept)
    consult = back.tool("billing", scope="acme")

    async def mixed(args: dict[str, Any]) -> str:
        asking = consult.dispatch("billing", {"input": "a"})
        if order == "after":
            await asking
            await asyncio.sleep(30)
        else:
            await asyncio.gather(asking, asyncio.sleep(30))
        return "never"

    outer = Script(use(ToolCall(id="m1", name="mixed", args={})), text_reply("never"))
    front = Engine(tree(tmp_path / "front", support=""), generation=outer, refinery=kept)
    stop = Stop()
    tools = Tools([Tool("mixed", "Consult, then work", body=mixed)])
    task = asyncio.create_task(front.run("support", "acme", "x", tools=tools, stop=stop))
    await until(lambda: billing.calls == 1, "the consult")
    began = time.monotonic()
    stop.now()
    run = await asyncio.wait_for(task, 2)
    assert time.monotonic() - began < 0.5
    assert isinstance(run, Generation) and run.stopped == "now"
    assert (run.steps[0].result, run.steps[0].stopped) == (STOPPED, True)
    [inner] = kept.of("billing")
    assert inner.stopped == "now"


async def test_one_press_cuts_the_rest_of_a_call_once(tmp_path: Path) -> None:
    billing = Gated()
    back = Engine(tree(tmp_path / "back", billing=""), generation=billing)
    consult = back.tool("billing", scope="acme")
    seen: list[int] = []

    async def work() -> None:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            running = asyncio.current_task()
            seen.append(0 if running is None else running.cancelling())
            raise

    async def mixed(args: dict[str, Any]) -> str:
        await asyncio.gather(consult.dispatch("billing", {"input": "a"}), work())
        return "never"

    outer = Script(use(ToolCall(id="m1", name="mixed", args={})), text_reply("never"))
    front = Engine(tree(tmp_path / "front", support=""), generation=outer)
    stop = Stop()
    tools = Tools([Tool("mixed", "Consult beside work", body=mixed)])
    task = asyncio.create_task(front.run("support", "acme", "x", tools=tools, stop=stop))
    await until(lambda: billing.calls == 1, "the consult")
    stop.now()
    run = await asyncio.wait_for(task, 2)
    assert isinstance(run, Generation) and run.stopped == "now"
    assert seen == [1]


@pytest.mark.parametrize("own", [True, False])
async def test_a_stop_before_a_call_returns_cuts_it_and_its_consult_keeps_its_record(
    tmp_path: Path, own: bool
) -> None:
    kept = Writing()
    billing = Script(text_reply("Refunded."))
    back = Engine(tree(tmp_path / "back", billing=""), generation=billing, refinery=kept)
    consult = back.tool("billing", scope="acme")

    async def asking(args: dict[str, Any]) -> Any:
        work = consult.dispatch("billing", {"input": "Was I charged twice?"})
        return await (work if own else asyncio.create_task(work))

    outer = Script(use(ToolCall(id="a1", name="ask", args={})), text_reply("never"))
    front = Engine(tree(tmp_path / "front", support=""), generation=outer, refinery=kept)
    stop = Stop()
    tools = Tools([Tool("ask", "Ask billing", body=asking)])
    task = asyncio.create_task(front.run("support", "acme", "x", tools=tools, stop=stop))
    await asyncio.wait_for(kept.writing.wait(), 2)
    stop.now()
    run = await asyncio.wait_for(task, 2)
    assert isinstance(run, Generation) and run.stopped == "now"
    assert (run.steps[0].result, run.steps[0].stopped) == (STOPPED, True)
    [inner] = kept.of("billing")
    assert inner.stopped is None and inner.reply == "Refunded."


class Hung(Kept):
    def __init__(self) -> None:
        super().__init__()
        self.writing = asyncio.Event()

    async def record(self, run: Run) -> None:
        self.writing.set()
        await asyncio.Event().wait()


async def test_a_tool_deadline_bounds_a_consult_whose_record_write_hangs(tmp_path: Path) -> None:
    back = Engine(
        tree(tmp_path / "back", billing=""), generation=Script(text_reply("ok")), refinery=Hung()
    )
    outer = Script(consulting(), text_reply("Carried on."))
    front = Engine(tree(tmp_path / "front", support=""), generation=outer, tool_timeout=0.2)
    began = time.monotonic()
    run = await asyncio.wait_for(
        front.run("support", "acme", "x", tools=back.tool("billing", scope="acme")), 2
    )
    assert time.monotonic() - began < 1.0
    assert isinstance(run, Generation) and run.reply == "Carried on."
    assert str(run.steps[0].result).startswith("TimeoutError")

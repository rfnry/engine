import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine import Stop
from rfnry_engine.engine import Engine
from rfnry_engine.models import Message, Reply, ToolCall, ToolSpec
from rfnry_engine.records import Event, Generation, Run
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import text_reply

STOPPED = "stopped by the user"


def tree(root: Path, **agents: str) -> Path:
    for name, front in agents.items():
        (root / name).mkdir(parents=True)
        (root / name / "AGENT.md").write_text(
            f"---\nname: {name}\nkind: generation\ndescription: {name.title()}\n{front}---\n"
            f"You are {name}.\n",
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

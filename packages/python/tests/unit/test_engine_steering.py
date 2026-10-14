import asyncio
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from rfnry_engine import Stop
from rfnry_engine.engine import Engine
from rfnry_engine.models import ChoiceAnswer, Message, Part, Reply, Text, ToolCall, ToolSpec
from rfnry_engine.records import Decision, Generation
from rfnry_engine.refinery.labels import Review
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.refinery.storage import Storage
from rfnry_engine.tools.tools import Tool, Tools
from rfnry_engine.tools.workspace import Local, Workspace

from .fakes import Fixed, said, text_reply

AGENTS = {
    "chat": "kind: generation\ndescription: Talks",
    "coder": "kind: generation\ndescription: Writes and runs code\nworkspace: true",
    "lead": "kind: generation\ndescription: Leads\nagents: [chat]",
    "desk": "kind: decision\ndescription: Sorts\nroute:\n  by: intent\n  to: {talk: chat}",
}
INTENT = (
    "---\ntype: choice\noptions:\n  talk: Just talk\n  other: Anything else\n---\nWhat is this?\n"
)
STOPPED = "stopped by the user"


async def until(check: Callable[[], bool], what: str, deadline: float = 10.0) -> None:
    try:
        async with asyncio.timeout(deadline):
            while not check():
                await asyncio.sleep(0.01)
    except TimeoutError:
        pytest.fail(f"{what} never happened")


def written(path: Path) -> bool:
    return path.exists() and bool(path.read_text().strip())


def make_tree(tmp_path: Path) -> Path:
    root = tmp_path / "agents"
    for name, front in AGENTS.items():
        (root / name).mkdir(parents=True)
        (root / name / "AGENT.md").write_text(
            f"---\nname: {name}\n{front}\n---\nYou are {name}.\n", encoding="utf-8"
        )
        if "kind: decision" in front:
            (root / name / "questions").mkdir()
            (root / name / "questions" / "intent.md").write_text(INTENT, encoding="utf-8")
    return root


def use(*calls: ToolCall) -> Reply:
    return Reply(text="", tool_calls=list(calls), stop_reason="tool_use")


def call(name: str, call_id: str, **args: Any) -> ToolCall:
    return ToolCall(id=call_id, name=name, args=args)


class Paced:
    def __init__(self, *script: Reply | tuple[float, Reply]) -> None:
        self.script = [item if isinstance(item, tuple) else (0.0, item) for item in script]
        self.calls: list[list[Message]] = []
        self.entered = asyncio.Event()

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        self.calls.append(list(messages))
        if not self.script:
            raise AssertionError("Paced ran out of replies")
        delay, reply = self.script.pop(0)
        self.entered.set()
        await asyncio.sleep(delay)
        return reply


class PacedStream(Paced):
    async def generate_stream(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        on_delta: Callable[[str], Awaitable[None]],
    ) -> Reply:
        self.calls.append(list(messages))
        delay, reply = self.script.pop(0)
        await on_delta("Let me ")
        await on_delta("look")
        self.entered.set()
        await asyncio.sleep(delay)
        return reply


class Slow:
    def __init__(self, seconds: float, result: str = "done") -> None:
        self.seconds = seconds
        self.result = result
        self.started = asyncio.Event()
        self.cancelled = False
        self.finished = False

    async def __call__(self, args: dict[str, Any]) -> str:
        self.started.set()
        try:
            await asyncio.sleep(self.seconds)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        self.finished = True
        return self.result


async def fast(args: dict[str, Any]) -> str:
    return "fast"


def answered(messages: list[Message]) -> bool:
    asked = [c.id for m in messages if m.role == "assistant" for c in m.tool_calls]
    results = [m.tool_call_id for m in messages if m.role == "tool"]
    return asked == results


async def test_stop_now_cancels_a_slow_model_call(tmp_path: Path, storage: Storage) -> None:
    llm = Paced((30.0, text_reply("never")))
    refinery = Refinery(storage)
    engine = Engine(make_tree(tmp_path), generation=llm, refinery=refinery)
    stop = Stop()
    task = asyncio.create_task(engine.run("chat", "acme", "hi", stop=stop))
    await llm.entered.wait()
    stop.now()
    run = await asyncio.wait_for(task, 2)
    assert isinstance(run, Generation)
    assert run.stopped == "now" and run.reply == "" and run.steps == [] and run.calls == 0
    assert run.conversation == [Message(role="user", content="hi")] and run.model is None
    assert await refinery.storage.get_run("acme", run.run_id) is not None


class Named(Paced):
    def __init__(self, model: object, *script: Reply | tuple[float, Reply]) -> None:
        super().__init__(*script)
        self.model = model


@pytest.mark.parametrize(("declared", "recorded"), [("m-1", "m-1"), ("", None), (7, None)])
async def test_a_run_stopped_in_its_first_call_records_the_model_its_adapter_declares(
    tmp_path: Path, declared: object, recorded: str | None
) -> None:
    llm = Named(declared, (30.0, text_reply("never")))
    stop = Stop()
    task = asyncio.create_task(
        Engine(make_tree(tmp_path), generation=llm).run("chat", "acme", "hi", stop=stop)
    )
    await llm.entered.wait()
    stop.now()
    run = await asyncio.wait_for(task, 2)
    assert isinstance(run, Generation) and run.stopped == "now" and run.calls == 0
    assert run.model == recorded


async def test_stop_now_keeps_the_streamed_text(tmp_path: Path) -> None:
    llm = PacedStream((30.0, text_reply("never")))
    engine = Engine(make_tree(tmp_path), generation=llm)
    stop = Stop()
    deltas: list[str] = []

    async def on_event(event: Any) -> None:
        if hasattr(event, "text"):
            deltas.append(event.text)

    task = asyncio.create_task(engine.run("chat", "acme", "hi", stop=stop, on_event=on_event))
    await llm.entered.wait()
    stop.now()
    run = await asyncio.wait_for(task, 2)
    assert isinstance(run, Generation)
    assert run.stopped == "now" and run.reply == "Let me look" and "".join(deltas) == "Let me look"
    assert run.conversation[-1] == Message(role="assistant", content="Let me look")


async def test_stop_now_cancels_a_slow_tool_and_answers_it(tmp_path: Path) -> None:
    slow = Slow(30.0)
    tools = Tools([Tool("slow", body=slow), Tool("fast", body=fast)])
    llm = Paced(use(call("slow", "c1"), call("fast", "c2")))
    engine = Engine(make_tree(tmp_path), generation=llm)
    stop = Stop()
    task = asyncio.create_task(engine.run("chat", "acme", "work", tools=tools, stop=stop))
    await slow.started.wait()
    stop.now()
    run = await asyncio.wait_for(task, 2)
    assert isinstance(run, Generation)
    assert run.stopped == "now" and slow.cancelled and not slow.finished
    assert [(s.tool, s.result, s.is_error, s.stopped) for s in run.steps] == [
        ("slow", STOPPED, True, True),
        ("fast", "fast", False, False),
    ]
    assert run.conversation[-2:] == [
        Message(role="tool", tool_call_id="c1", content=STOPPED, is_error=True),
        Message(role="tool", tool_call_id="c2", content="fast"),
    ]
    assert answered(run.conversation) and len(llm.calls) == 1


async def test_after_step_lets_the_running_tool_finish_then_ends(tmp_path: Path) -> None:
    slow = Slow(0.2)
    llm = Paced(
        Reply(text="Working on it.", tool_calls=[call("slow", "c1")], stop_reason="tool_use")
    )
    engine = Engine(make_tree(tmp_path), generation=llm)
    stop = Stop()
    task = asyncio.create_task(
        engine.run("chat", "acme", "work", tools=Tools([Tool("slow", body=slow)]), stop=stop)
    )
    await slow.started.wait()
    stop.after_step()
    run = await asyncio.wait_for(task, 2)
    assert isinstance(run, Generation)
    assert run.stopped == "after_step" and slow.finished and not slow.cancelled
    assert [(s.result, s.is_error, s.stopped) for s in run.steps] == [("done", False, False)]
    assert run.reply == "Working on it." and len(llm.calls) == 1 and answered(run.conversation)


async def test_after_step_during_a_final_reply_changes_nothing(tmp_path: Path) -> None:
    llm = Paced((0.1, text_reply("All done.")))
    engine = Engine(make_tree(tmp_path), generation=llm)
    stop = Stop()
    task = asyncio.create_task(engine.run("chat", "acme", "hi", stop=stop))
    await llm.entered.wait()
    stop.after_step()
    run = await task
    assert isinstance(run, Generation) and run.stopped is None and run.reply == "All done."


async def test_now_after_after_step_upgrades_the_stop(tmp_path: Path) -> None:
    slow = Slow(30.0)
    llm = Paced(use(call("slow", "c1")))
    engine = Engine(make_tree(tmp_path), generation=llm)
    stop = Stop()
    task = asyncio.create_task(
        engine.run("chat", "acme", "x", tools=Tools([Tool("slow", body=slow)]), stop=stop)
    )
    await slow.started.wait()
    stop.after_step()
    stop.now()
    stop.after_step()
    run = await asyncio.wait_for(task, 2)
    assert isinstance(run, Generation) and run.stopped == "now" and slow.cancelled


async def test_a_stopped_run_is_recorded_and_continues_cleanly(
    tmp_path: Path, storage: Storage
) -> None:
    slow = Slow(30.0)
    llm = Paced(use(call("slow", "c1")), text_reply("Picking up where we left off."))
    refinery = Refinery(storage)
    engine = Engine(make_tree(tmp_path), generation=llm, refinery=refinery)
    tools = Tools([Tool("slow", body=slow)])
    stop = Stop()
    task = asyncio.create_task(engine.run("chat", "acme", "work", tools=tools, stop=stop))
    await slow.started.wait()
    stop.now()
    run = await asyncio.wait_for(task, 2)
    assert isinstance(run, Generation)
    row = await refinery.storage.get_run("acme", run.run_id)
    assert row is not None
    stored = Generation.model_validate(row.body)
    assert stored.stopped == "now" and stored.steps[0].stopped
    await refinery.review(
        "acme", run.run_id, Review(stance="objection", correction="Do it differently.")
    )
    after = await engine.run("chat", "acme", "go on", history=run.conversation, tools=tools)
    assert isinstance(after, Generation) and after.stopped is None
    sent = llm.calls[1]
    assert answered(sent) and sent[-1] == Message(role="user", content="go on")


async def test_a_stop_before_the_run_ends_it_before_the_first_model_call(tmp_path: Path) -> None:
    llm = Paced(text_reply("never"))
    engine = Engine(make_tree(tmp_path), generation=llm)
    history = [Message(role="user", content="a"), Message(role="assistant", content="b")]
    for mode in ("now", "after_step"):
        stop = Stop()
        getattr(stop, mode)()
        run = await engine.run("chat", "acme", "hi", history=history, stop=stop)
        assert isinstance(run, Generation)
        assert run.stopped == mode and run.calls == 0 and run.reply == ""
        assert run.conversation == [*history, Message(role="user", content="hi")]
    assert llm.calls == []


async def test_a_stop_after_the_run_is_harmless(tmp_path: Path) -> None:
    llm = Paced(text_reply("Done."))
    engine = Engine(make_tree(tmp_path), generation=llm)
    stop = Stop()
    run = await engine.run("chat", "acme", "hi", stop=stop)
    stop.now()
    stop.after_step()
    assert isinstance(run, Generation) and run.stopped is None and run.reply == "Done."


async def test_inbox_items_land_at_the_next_boundary_in_order(tmp_path: Path) -> None:
    inbox: asyncio.Queue[Any] = asyncio.Queue()
    parts: list[Part] = [Text(text="and this")]

    async def sending(args: dict[str, Any]) -> str:
        inbox.put_nowait("also check the tests")
        inbox.put_nowait(parts)
        return "read"

    llm = Paced(use(call("send", "c1")), text_reply("Checked both."))
    engine = Engine(make_tree(tmp_path), generation=llm)
    inbox.put_nowait("queued before")
    run = await engine.run(
        "chat", "acme", "work", tools=Tools([Tool("send", body=sending)]), inbox=inbox
    )
    assert isinstance(run, Generation)
    assert llm.calls[0] == [Message(role="user", content="work")]
    assert llm.calls[1][-4:] == [
        Message(role="tool", tool_call_id="c1", content="read"),
        Message(role="user", content="queued before"),
        Message(role="user", content="also check the tests"),
        Message(role="user", content=parts),
    ]
    assert said(run.conversation[:-1]) == llm.calls[1] and run.messages == llm.calls[0]
    assert inbox.empty()


async def test_inbox_leftovers_stay_in_the_queue(tmp_path: Path) -> None:
    inbox: asyncio.Queue[Any] = asyncio.Queue()
    llm = Paced((0.1, text_reply("Done.")))
    engine = Engine(make_tree(tmp_path), generation=llm)
    task = asyncio.create_task(engine.run("chat", "acme", "hi", inbox=inbox))
    await llm.entered.wait()
    inbox.put_nowait("one more thing")
    run = await task
    assert isinstance(run, Generation) and run.reply == "Done."
    assert all(m.content != "one more thing" for m in run.conversation)
    assert inbox.get_nowait() == "one more thing"


async def test_inbox_with_after_step_pending_drains_nothing(tmp_path: Path) -> None:
    inbox: asyncio.Queue[Any] = asyncio.Queue()
    stop = Stop()

    async def sending(args: dict[str, Any]) -> str:
        inbox.put_nowait("wait")
        stop.after_step()
        return "ok"

    llm = Paced(use(call("send", "c1")))
    engine = Engine(make_tree(tmp_path), generation=llm)
    run = await engine.run(
        "chat", "acme", "x", tools=Tools([Tool("send", body=sending)]), inbox=inbox, stop=stop
    )
    assert isinstance(run, Generation) and run.stopped == "after_step"
    assert run.conversation[-1].role == "tool" and inbox.get_nowait() == "wait"


async def test_a_delegated_child_observes_the_stop(tmp_path: Path) -> None:
    llm = Paced(use(call("Delegate", "d1", agent="chat", input="go")), (30.0, text_reply("never")))
    engine = Engine(make_tree(tmp_path), generation=llm)
    stop = Stop()
    inbox: asyncio.Queue[Any] = asyncio.Queue()
    task = asyncio.create_task(engine.run("lead", "acme", "x", stop=stop, inbox=inbox))
    while len(llm.calls) < 2:
        await asyncio.sleep(0.01)
    inbox.put_nowait("for the lead")
    stop.now()
    run = await asyncio.wait_for(task, 2)
    assert isinstance(run, Generation) and run.stopped == "now"
    child = run.children[0]
    assert (
        isinstance(child, Generation)
        and child.stopped == "now"
        and child.parent_run_id == run.run_id
    )
    step = run.steps[0]
    assert (step.result, step.is_error, step.stopped, step.child_run_id) == (
        STOPPED,
        True,
        True,
        child.run_id,
    )
    assert inbox.get_nowait() == "for the lead"


async def test_a_routed_run_shares_the_stop_and_the_inbox(tmp_path: Path) -> None:
    decide = Fixed(
        {"intent": ChoiceAnswer(choice="talk", probabilities={"talk": 0.9, "other": 0.1})}
    )
    llm = Paced(use(call("fast", "c1")), text_reply("ok"))
    engine = Engine(make_tree(tmp_path), generation=llm, decision=decide)
    inbox: asyncio.Queue[Any] = asyncio.Queue()
    inbox.put_nowait("steer")
    run = await engine.run(
        "desk", "acme", "hi", tools=Tools([Tool("fast", body=fast)]), inbox=inbox
    )
    assert isinstance(run, Decision) and isinstance(run.routed_to, Generation)
    assert run.routed_to.conversation[-2] == Message(role="user", content="steer")
    stop = Stop()
    stop.now()
    stopped = await engine.run("desk", "acme", "hi", stop=stop)
    assert isinstance(stopped, Decision) and isinstance(stopped.routed_to, Generation)
    assert stopped.routed_to.stopped == "now" and stopped.routed_to.calls == 0


async def test_stop_now_kills_a_workspace_shell(tmp_path: Path) -> None:
    root = tmp_path / "work"
    root.mkdir()
    work = Workspace(Local(root), scope="acme")
    llm = Paced(use(call("Shell", "c1", command="sleep 30 & echo $! > child.pid; wait")))
    engine = Engine(make_tree(tmp_path), generation=llm, tool_timeout=None)
    stop = Stop()
    task = asyncio.create_task(engine.run("coder", "acme", "x", workspace=work, stop=stop))
    try:
        pidfile = root / "child.pid"
        await until(lambda: written(pidfile), "the shell writing its child's pid")
        pid = int(pidfile.read_text())
        stop.now()
        run = await asyncio.wait_for(task, 10)
    finally:
        task.cancel()
        await asyncio.wait([task])
    assert isinstance(run, Generation) and run.stopped == "now" and run.steps[0].result == STOPPED

    def gone() -> bool:
        try:
            os.kill(pid, 0)
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except (ProcessLookupError, FileNotFoundError):
            return True
        return state == "Z"

    await until(gone, "the shell's child stopping")


async def test_stop_now_sends_a_workspace_command_sigterm_first(tmp_path: Path) -> None:
    root = tmp_path / "work"
    root.mkdir()
    command = "trap 'echo stopped > stopped.txt; exit 0' TERM; touch ready; sleep 30 & wait"
    llm = Paced(use(call("Shell", "c1", command=command)))
    engine = Engine(make_tree(tmp_path), generation=llm, tool_timeout=None)
    stop = Stop()
    task = asyncio.create_task(
        engine.run("coder", "acme", "x", workspace=Workspace(Local(root), scope="acme"), stop=stop)
    )
    try:
        await until((root / "ready").exists, "the shell starting")
        stop.now()
        run = await asyncio.wait_for(task, 10)
    finally:
        task.cancel()
        await asyncio.wait([task])
    assert isinstance(run, Generation) and run.stopped == "now" and run.steps[0].result == STOPPED
    assert (root / "stopped.txt").read_text() == "stopped\n"


class Verdict(BaseModel):
    ok: bool


async def test_stop_now_answers_a_pending_output_call(tmp_path: Path) -> None:
    slow = Slow(30.0)
    llm = Paced(use(call("slow", "c1"), call("Output", "o1", ok=True)))
    engine = Engine(make_tree(tmp_path), generation=llm)
    stop = Stop()
    tools = Tools([Tool("slow", body=slow)])
    task = asyncio.create_task(
        engine.run("chat", "acme", "x", tools=tools, expect=Verdict, stop=stop)
    )
    await slow.started.wait()
    stop.now()
    run = await asyncio.wait_for(task, 2)
    assert isinstance(run, Generation) and run.stopped == "now" and run.reply == ""
    assert [(s.tool, s.stopped) for s in run.steps] == [("slow", True), ("Output", True)]
    assert answered(run.conversation)


async def test_stop_now_answers_the_calls_after_a_pending_output_too(tmp_path: Path) -> None:
    slow = Slow(30.0)
    llm = Paced(use(call("slow", "c1"), call("Output", "o1", ok=True), call("LS", "l1")))
    engine = Engine(make_tree(tmp_path), generation=llm)
    stop = Stop()
    task = asyncio.create_task(
        engine.run(
            "chat", "acme", "x", tools=Tools([Tool("slow", body=slow)]), expect=Verdict, stop=stop
        )
    )
    await slow.started.wait()
    stop.now()
    run = await asyncio.wait_for(task, 2)
    assert isinstance(run, Generation) and run.stopped == "now"
    assert [(s.tool, s.stopped) for s in run.steps] == [
        ("slow", True),
        ("Output", True),
        ("LS", True),
    ]
    assert answered(run.conversation)


async def test_stop_now_leaves_a_workspace_job_running(tmp_path: Path) -> None:
    root = tmp_path / "work"
    root.mkdir()
    work = Workspace(Local(root), scope="acme")
    llm = Paced(
        use(call("Shell", "c1", command="sleep 30 & echo $! > job.pid; wait", background=True)),
        use(call("Shell", "c2", command="touch ready; sleep 30")),
    )
    engine = Engine(make_tree(tmp_path), generation=llm, tool_timeout=None)
    stop = Stop()
    task = asyncio.create_task(engine.run("coder", "acme", "x", workspace=work, stop=stop))
    try:
        await until((root / "ready").exists, "the second command starting")
        await until(lambda: written(root / "job.pid"), "the job writing its pid")
        stop.now()
        run = await asyncio.wait_for(task, 10)
        assert isinstance(run, Generation) and run.stopped == "now"
        os.kill(int((root / "job.pid").read_text()), 0)
        jobs = next(t for t in work.tools() if t.name == "Jobs")
        assert [listed["running"] for listed in (await jobs.call({}))["jobs"]] == [True]
    finally:
        await work.close()

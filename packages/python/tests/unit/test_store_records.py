import asyncio
import json
import logging
from collections.abc import Collection
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from rfnry_engine.models import Message, Reply, ToolCall
from rfnry_engine.records import Decision, Generation, Run
from rfnry_engine.refinery.postgres import PostgresStorage
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.refinery.storage import RunRow
from rfnry_engine.stop import Stop

from .fakes import (
    Crew,
    Ledger,
    MemoryBucket,
    MemoryStore,
    background,
    crew,
    delegating,
    generation_record,
    released,
    team,
    text_reply,
    until,
)
from .test_background import pending
from .test_store import Failing, exactly
from .test_store_tree import jobs, lead
from .trees import write_folder

ROUTED = '{"type": "choice", "instructions": "Which?", "options": {"talk": null, "work": null}}'


class Logged:
    def __init__(self) -> None:
        self.log: list[tuple[str, list[str]]] = []
        self.given: list[Collection[str]] = []

    async def record(self, run: Run, *, skip: Collection[str] = ()) -> None:
        self.log.append((run.run_id, sorted(skip)))
        self.given.append(skip)


class Plain:
    def __init__(self) -> None:
        self.runs: list[Run] = []

    async def record(self, run: Run) -> None:
        self.runs.append(run)


class Holding:
    def __init__(self, agent: str, first: bool = False) -> None:
        self.agent, self.first = agent, first
        self.runs: list[Run] = []
        self.writing, self.released = asyncio.Event(), asyncio.Event()
        self.cut = 0

    async def record(self, run: Run, *, skip: Collection[str] = ()) -> None:
        if run.agent == self.agent and not (self.first and self.writing.is_set()):
            self.writing.set()
            try:
                await self.released.wait()
            except asyncio.CancelledError:
                self.cut += 1
                raise
        self.runs.append(run)


class Counted(PostgresStorage):
    def __init__(self, pool: Any, failing: str | None = None) -> None:
        super().__init__(pool)
        self.writes: list[list[str]] = []
        self.failing = failing

    async def put_runs(self, scope: str, rows: list[RunRow]) -> None:
        if rows and rows[0].agent == self.failing:
            self.failing = None
            raise OSError("storage is down")
        self.writes.append([row.run_id for row in rows])
        await super().put_runs(scope, rows)


def sorting(text: str = "sort it") -> Reply:
    call = ToolCall(
        id="b2", name="Delegate", args={"agent": "desk", "input": text, "background": True}
    )
    return Reply(tool_calls=[call], stop_reason="tool_use")


class Done(BaseModel):
    done: bool


def routing(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return write_folder(
        root,
        {
            "lead": {
                "kind": "generation",
                "model": "crew",
                "description": "Leads",
                "instructions": "lead.md",
                "orchestration": {"delegate_to": ["router", "worker"], "background": True},
            },
            "router": {
                "kind": "decision",
                "model": "judge",
                "description": "Routes",
                "instructions": "router.md",
                "questions": ["router/*.json"],
                "orchestration": {"route": {"by": "intent", "to": {"work": "worker"}}},
            },
            "worker": {
                "kind": "generation",
                "model": "crew",
                "description": "Works",
                "instructions": "worker.md",
                "orchestration": {"delegate_to": ["deep"], "background": True},
            },
            "deep": {
                "kind": "generation",
                "model": "crew",
                "description": "Digs",
                "instructions": "deep.md",
            },
        },
        files={
            "lead.md": "Lead.\n",
            "router.md": "",
            "worker.md": "Work.\n",
            "deep.md": "Dig.\n",
            "router/intent.json": ROUTED,
        },
    )


def results(messages: list[Message]) -> dict[str, Message]:
    return {str(message.tool_call_id): message for message in messages if message.role == "tool"}


def exported(bucket: MemoryBucket) -> list[str]:
    return [
        json.loads(line)["run_id"]
        for data in bucket.objects.values()
        for line in data.decode().splitlines()
    ]


async def test_a_store_run_records_each_helper_when_it_ends(tmp_path: Path) -> None:
    gate, logged = asyncio.Event(), Logged()
    first = Reply(
        tool_calls=[*background("part one").tool_calls, *sorting().tool_calls],
        stop_reason="tool_use",
    )
    model = Crew([first, gate, *[text_reply("waiting")] * 3], {"part one": [text_reply("one")]})
    engine = team(crew(tmp_path, background=True), model, refinery=logged)
    work = asyncio.create_task(lead(engine, MemoryStore()))
    await until(lambda: len(logged.log) == 2, "the helpers' records")
    run = await released(work, model, gate)
    helped = jobs(run)
    (one, before), (other, after), last = logged.log
    assert {one, other} == {helped["d1"].run_id, helped["d2"].run_id}
    assert before == [] and after == [one]
    assert last == (run.run_id, sorted([one, other]))
    assert [sorted(skip) for skip in logged.given] == [skip for _, skip in logged.log]


async def test_a_synchronous_child_is_recorded_when_it_ends(tmp_path: Path) -> None:
    gate, logged = asyncio.Event(), Logged()
    model = Crew(
        [delegating("worker", "part one"), gate, text_reply("done")],
        {"part one": [text_reply("one")]},
    )
    engine = team(crew(tmp_path), model, refinery=logged)
    work = asyncio.create_task(lead(engine, MemoryStore()))
    await until(lambda: len(model.calls["lead"]) == 2, "the lead's call 2")
    assert len(logged.log) == 1 and not work.done()
    gate.set()
    run = await asyncio.wait_for(work, 5)
    [child] = run.children
    assert logged.log == [(child.run_id, []), (run.run_id, [child.run_id])]


async def test_every_row_of_a_store_run_is_written_once(
    tmp_path: Path, pool: Any, storage: PostgresStorage
) -> None:
    counted = Counted(pool)
    refinery = Refinery(counted)
    model = Crew(
        [background("part one", "part two"), *[text_reply("waiting")] * 3],
        {
            "part one": [background("deep"), *[text_reply("waiting below")] * 3],
            "part two": [text_reply("two")],
            "deep": [text_reply("found")],
        },
    )
    worker = {"orchestration": {"delegate_to": ["worker"], "background": True}}
    engine = team(crew(tmp_path, worker=worker, background=True), model, refinery=refinery)
    run = await lead(engine, MemoryStore())
    captain = jobs(run)["d1"]
    assert isinstance(captain, Generation)
    tree = [run.run_id, captain.run_id, jobs(captain)["d1"].run_id, jobs(run)["d2"].run_id]
    written = [run_id for rows in counted.writes for run_id in rows]
    assert sorted(written) == sorted(tree)
    assert [len(rows) for rows in counted.writes] == [1, 1, 1, 1]
    assert counted.writes[-1] == [run.run_id]
    bucket = MemoryBucket()
    await refinery.export("acme", bucket, since=None)
    assert sorted(exported(bucket)) == sorted(tree)


async def test_a_helper_whose_record_failed_is_written_with_its_lead(
    tmp_path: Path, pool: Any, storage: PostgresStorage, caplog: pytest.LogCaptureFixture
) -> None:
    counted = Counted(pool, failing="worker")
    refinery = Refinery(counted)
    model = Crew(
        [background("part one"), *[text_reply("waiting")] * 3], {"part one": [text_reply("one")]}
    )
    engine = team(crew(tmp_path, background=True), model, refinery=refinery)
    with caplog.at_level(logging.WARNING, logger="rfnry_engine.engine"):
        run = await lead(engine, MemoryStore())
    helper = jobs(run)["d1"]
    assert counted.writes == [[run.run_id, helper.run_id]]
    [line] = [record for record in caplog.records if record.name == "rfnry_engine.engine"]
    assert line.levelno == logging.WARNING
    assert line.getMessage() == f"engine: refinery.record failed for run {helper.run_id}"
    assert (await refinery.run("acme", helper.run_id)).run_id == helper.run_id


async def test_a_failed_store_run_writes_each_row_once(
    tmp_path: Path, pool: Any, storage: PostgresStorage
) -> None:
    counted, gate = Counted(pool), asyncio.Event()
    model = Failing([background("look")], {"look": [gate, text_reply("never")]})
    engine = team(crew(tmp_path, background=True), model, refinery=Refinery(counted))
    with pytest.raises(ValueError, match=exactly("the lead's model broke")):
        await lead(engine, MemoryStore())
    [helper], [failure] = counted.writes
    back = await Refinery(counted).run("acme", failure)
    assert isinstance(back, Generation) and back.error is not None
    assert [child.run_id for child in back.children] == [helper]


async def test_without_a_store_helpers_are_recorded_with_their_lead_only(tmp_path: Path) -> None:
    plain = Plain()
    model = Crew(
        [background("part one", "part two"), *[text_reply("waiting")] * 3],
        {"part one": [text_reply("one")], "part two": [text_reply("two")]},
    )
    engine = team(crew(tmp_path, background=True), model, refinery=plain)
    run = await engine.run("lead", "acme", "go")
    assert len(plain.runs) == 1 and plain.runs[0] is run
    assert isinstance(run, Generation) and len(run.children) == 2


async def test_record_false_records_nothing_in_store_mode(tmp_path: Path) -> None:
    logged = Logged()
    model = Crew(
        [
            background("part one"),
            delegating("worker", "part two", call_id="s1"),
            *[text_reply("waiting")] * 3,
        ],
        {"part one": [text_reply("one")], "part two": [text_reply("two")]},
    )
    engine = team(crew(tmp_path, background=True), model, refinery=logged)
    run = await lead(engine, MemoryStore(), record=False)
    assert logged.log == [] and len(run.children) == 2


@pytest.mark.parametrize("who", ["synchronous child", "helper"])
async def test_a_stop_pressed_while_a_child_is_recorded_keeps_its_record(
    who: str, tmp_path: Path
) -> None:
    stop, holding = Stop(), Holding("worker")
    started = delegating("worker", "part one") if who != "helper" else background("part one")
    model = Crew([started, *[text_reply("waiting")] * 3], {"part one": [text_reply("one")]})
    engine = team(crew(tmp_path, background=who == "helper"), model, refinery=holding)
    work = asyncio.create_task(lead(engine, MemoryStore(), stop=stop))
    await asyncio.wait_for(holding.writing.wait(), 5)
    stop.now()
    for _ in range(20):
        await asyncio.sleep(0)
    assert not work.done()
    holding.released.set()
    run = await asyncio.wait_for(work, 5)
    [child] = run.children
    assert run.stopped == "now" and holding.cut == 0
    assert holding.runs == [child, run] and child.error is None


async def test_a_routing_helper_s_routed_run_is_written_once(
    tmp_path: Path, pool: Any, storage: PostgresStorage
) -> None:
    counted = Counted(pool)
    model = Crew(
        [background("part one", agent="router"), *[text_reply("waiting")] * 3],
        {
            "part one": [background("deep", agent="deep"), *[text_reply("waiting below")] * 3],
            "deep": [text_reply("found")],
        },
    )
    engine = team(routing(tmp_path), model, refinery=Refinery(counted))
    run = await lead(engine, MemoryStore())
    [router] = run.children
    assert isinstance(router, Decision) and isinstance(router.routed_to, Generation)
    [deep] = router.routed_to.children
    assert counted.writes == [
        [deep.run_id],
        [router.run_id, router.routed_to.run_id],
        [run.run_id],
    ]


async def test_a_row_whose_record_failed_is_written_once_with_the_run_above_it(
    tmp_path: Path, pool: Any, storage: PostgresStorage
) -> None:
    counted = Counted(pool, failing="deep")
    model = Crew(
        [background("part one"), *[text_reply("waiting")] * 3],
        {
            "part one": [background("deep", agent="deep"), *[text_reply("waiting below")] * 3],
            "deep": [text_reply("found")],
        },
    )
    engine = team(routing(tmp_path), model, refinery=Refinery(counted))
    run = await lead(engine, MemoryStore())
    captain = jobs(run)["d1"]
    assert isinstance(captain, Generation)
    deep = jobs(captain)["d1"]
    assert counted.writes == [[captain.run_id, deep.run_id], [run.run_id]]


async def test_a_failed_helper_is_recorded_when_it_ends(
    tmp_path: Path, pool: Any, storage: PostgresStorage
) -> None:
    counted, gate = Counted(pool), asyncio.Event()
    model = Crew([background("part one"), gate, *[text_reply("waiting")] * 3], {"part one": []})
    engine = team(crew(tmp_path, background=True), model, refinery=Refinery(counted))
    work = asyncio.create_task(lead(engine, MemoryStore()))
    await until(lambda: len(counted.writes) == 1, "the helper's record")
    run = await released(work, model, gate)
    helper = jobs(run)["d1"]
    assert helper.error is not None
    assert counted.writes == [[helper.run_id], [run.run_id]]


async def test_a_failed_synchronous_child_is_recorded_when_it_ends(
    tmp_path: Path, pool: Any, storage: PostgresStorage
) -> None:
    counted, gate = Counted(pool), asyncio.Event()
    model = Crew([delegating("worker", "part one"), gate, text_reply("done")], {"part one": []})
    engine = team(crew(tmp_path), model, refinery=Refinery(counted))
    work = asyncio.create_task(lead(engine, MemoryStore()))
    await until(lambda: len(model.calls["lead"]) == 2, "the lead's call 2")
    assert len(counted.writes) == 1 and not work.done()
    gate.set()
    run = await asyncio.wait_for(work, 5)
    [child] = run.children
    assert child.error is not None
    assert counted.writes == [[child.run_id], [run.run_id]]


async def test_a_helper_reads_as_ended_while_its_record_is_written(
    tmp_path: Path, pool: Any, storage: PostgresStorage
) -> None:
    ledger, gate, later = Ledger(pool, "worker"), asyncio.Event(), asyncio.Event()
    asking = Reply(
        tool_calls=[
            ToolCall(id="j1", name="Job", args={"id": "d1", "send": "one more thing"}),
            ToolCall(id="j2", name="Job", args={"id": "d1"}),
            ToolCall(id="j3", name="Job", args={"id": "d1", "stop": True}),
            *background("part two", prefix="c").tool_calls,
        ],
        stop_reason="tool_use",
    )
    model = Crew(
        [background("part one"), gate, asking, later, *[text_reply("waiting")] * 3],
        {"part one": [text_reply("one")], "part two": [text_reply("two")]},
    )
    folder = crew(tmp_path, background=True, max_background=1)
    engine = team(folder, model, refinery=Refinery(ledger))
    work = asyncio.create_task(lead(engine, MemoryStore()))
    await asyncio.wait_for(ledger.writing.wait(), 5)
    gate.set()
    await until(lambda: len(model.calls["lead"]) == 3, "the lead's call 3")
    assert not ledger.written.is_set()
    answered = results(model.calls["lead"][2])
    assert answered["j1"].is_error and answered["j1"].content == (
        "InputError: tools/helpers: helper d1 has ended; send reaches only a helper still running"
    )
    ended = {"running": False, "agent": "worker", "asking": None, "reply": "one"}
    assert json.loads(str(answered["j2"].content)) == ended
    assert json.loads(str(answered["j3"].content)) == ended
    assert not answered["c1"].is_error and answered["c1"].content == '{"job": "d2"}'
    ledger.released.set()
    later.set()
    run = await asyncio.wait_for(work, 5)
    assert sorted(jobs(run)) == ["d1", "d2"]


async def test_an_output_waits_only_for_the_record_of_a_helper_that_ended(
    tmp_path: Path, pool: Any, storage: PostgresStorage
) -> None:
    ledger, gate = Ledger(pool, "worker"), asyncio.Event()
    output = Reply(
        tool_calls=[ToolCall(id="o1", name="Output", args={"done": True})], stop_reason="tool_use"
    )
    model = Crew([background("part one"), gate, output], {"part one": [text_reply("one")]})
    engine = team(crew(tmp_path, background=True), model, refinery=Refinery(ledger))
    work = asyncio.create_task(lead(engine, MemoryStore(), expect=Done))
    await asyncio.wait_for(ledger.writing.wait(), 5)
    gate.set()
    for _ in range(20):
        await asyncio.sleep(0)
    assert not work.done() and len(model.calls["lead"]) == 2
    ledger.released.set()
    run = await asyncio.wait_for(work, 5)
    assert run.steps[-1].tool == "Output" and run.steps[-1].result == "accepted"
    notice = run.conversation[-1].notice
    assert notice is not None and (notice.job, notice.kind) == ("d1", "finished")
    assert run.conversation[-2].content == "accepted"
    assert [child.run_id for child in run.children] == [jobs(run)["d1"].run_id]
    assert (await Refinery(ledger).run("acme", run.run_id)).run_id == run.run_id


async def test_a_record_whose_rows_are_all_skipped_calls_no_storage(
    pool: Any, storage: PostgresStorage
) -> None:
    counted = Counted(pool)
    child = generation_record(agent="worker")
    run = generation_record(children=[child])
    await Refinery(counted).record(run, skip={run.run_id, child.run_id})
    assert counted.writes == []
    await Refinery(counted).record(run, skip={child.run_id})
    assert counted.writes == [[run.run_id]]


async def test_with_a_store_helpers_join_children_in_the_order_their_lead_hears_them(
    tmp_path: Path,
) -> None:
    holding, second, gate = Holding("worker", first=True), asyncio.Event(), asyncio.Event()
    model = Crew(
        [background("part one", "part two"), gate, *[text_reply("waiting")] * 4],
        {"part one": [text_reply("one")], "part two": [second, text_reply("two")]},
    )
    engine = team(crew(tmp_path, background=True), model, refinery=holding)
    work = asyncio.create_task(lead(engine, MemoryStore()))
    await asyncio.wait_for(holding.writing.wait(), 5)
    second.set()
    await until(lambda: len(holding.runs) == 1, "the second helper's record")
    holding.released.set()
    await until(lambda: len(holding.runs) == 2, "the first helper's record")
    gate.set()
    run = await asyncio.wait_for(work, 5)
    named = {child.run_id: job for job, child in jobs(run).items()}
    joined = [named[child.run_id] for child in run.children]
    heard = [message.notice.job for message in run.conversation if message.notice is not None]
    assert joined == heard == ["d2", "d1"]


async def test_job_stop_returns_once_the_helper_s_run_ends_while_its_record_is_written(
    tmp_path: Path,
) -> None:
    holding, never = Holding("worker"), asyncio.Event()
    stopping = Reply(
        tool_calls=[ToolCall(id="j1", name="Job", args={"id": "d1", "stop": True})],
        stop_reason="tool_use",
    )
    model = Crew(
        [background("part one"), stopping, *[text_reply("waiting")] * 4],
        {"part one": [never, text_reply("never")]},
    )
    engine = team(crew(tmp_path, background=True), model, refinery=holding)
    work = asyncio.create_task(lead(engine, MemoryStore()))
    await asyncio.wait_for(holding.writing.wait(), 5)
    await until(lambda: len(model.calls["lead"]) == 3, "the lead's call 3")
    assert holding.runs == []
    read = json.loads(str(results(model.calls["lead"][2])["j1"].content))
    assert read["running"] is False and "stopped" in read
    holding.released.set()
    run = await asyncio.wait_for(work, 5)
    helper = jobs(run)["d1"]
    assert isinstance(helper, Generation) and helper.stopped == "now"


async def test_a_cancelled_lead_cuts_a_helper_s_record_in_flight(tmp_path: Path) -> None:
    holding = Holding("worker")
    model = Crew(
        [background("part one"), *[text_reply("waiting")] * 3], {"part one": [text_reply("one")]}
    )
    engine = team(crew(tmp_path, background=True), model, refinery=holding)
    before = asyncio.all_tasks()
    work = asyncio.create_task(lead(engine, MemoryStore()))
    await asyncio.wait_for(holding.writing.wait(), 5)
    for _ in range(20):
        await asyncio.sleep(0)
    work.cancel()
    with pytest.raises(asyncio.CancelledError):
        await work
    assert holding.cut == 1 and holding.runs == [] and pending(before) == []

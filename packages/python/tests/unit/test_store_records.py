import asyncio
import json
import logging
from collections.abc import Collection
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.models import Reply, ToolCall
from rfnry_engine.records import Generation, Run
from rfnry_engine.refinery.postgres import PostgresStorage
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.refinery.storage import RunRow
from rfnry_engine.stop import Stop

from .fakes import (
    Crew,
    MemoryBucket,
    MemoryStore,
    background,
    crew,
    delegating,
    released,
    team,
    text_reply,
    until,
)
from .test_store import Failing, exactly
from .test_store_tree import jobs, lead


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
    def __init__(self, agent: str) -> None:
        self.agent = agent
        self.runs: list[Run] = []
        self.writing, self.released = asyncio.Event(), asyncio.Event()
        self.cut = 0

    async def record(self, run: Run, *, skip: Collection[str] = ()) -> None:
        if run.agent == self.agent:
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
        if rows[0].agent == self.failing:
            self.failing = None
            raise OSError("storage is down")
        self.writes.append([row.run_id for row in rows])
        await super().put_runs(scope, rows)


def sorting(text: str = "sort it") -> Reply:
    call = ToolCall(
        id="b2", name="Delegate", args={"agent": "desk", "input": text, "background": True}
    )
    return Reply(tool_calls=[call], stop_reason="tool_use")


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

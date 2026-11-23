import asyncio
import re
import secrets
from collections import Counter

import pytest
from rfnry_engine import Message, Notice, PostgresStorage

import harness
import measure
import merge
import moon
from board import Board, write_note
from fakes import FakeGate
from merge import Price, cost_of
from store import Store

PRICE = Price(10.0, 10.0, 10.0, 10.0, 100_000)
ENDED = 120


def fresh():
    return f"t-{secrets.token_hex(4)}"


async def ended(name):
    return await asyncio.wait_for(harness.wait(name), ENDED)


async def stored(pool, name):
    store = Store(pool)
    rows = await pool.fetch(
        "select distinct conversation from messages where scope = 'moon'"
        " and (conversation = $1 or starts_with(conversation, $1 || '/'))",
        name,
    )
    return {row["conversation"]: await store.read("moon", row["conversation"], 0) for row in rows}


def replies(messages):
    return [m for m in messages if m.role == "assistant" and m.usage is not None]


def prompts(messages):
    return [m.usage.input + m.usage.cache_read + m.usage.cache_creation for m in replies(messages)]


async def records_of(engine_pool, name):
    return await PostgresStorage(engine_pool).run_tree("moon", tags={"survey": name})


async def test_measure_reads_the_numbers_of_a_survey(service, pool, engine_pool, monkeypatch):
    monkeypatch.setitem(merge.PRICES, "fake", PRICE)
    await service(size=1, per_minute=6000)
    name = fresh()
    await harness.run(name, mode="fake", shape="crews", size=10, cap=100.0)
    assert (await ended(name))["status"] == "done"

    found = await measure.measure(pool, engine_pool, name)
    kept = await stored(pool, name)
    records = await records_of(engine_pool, name)
    assert found["survey"] == name and found["label"] == "fake"
    assert (found["mode"], found["shape"], found["status"]) == ("fake", "crews", "done")
    assert found["usage"] == "characters / 4, the fakes' estimate"
    assert (found["regions"], found["landed"], found["failed"], found["landed_twice"]) == (
        10,
        10,
        0,
        0,
    )

    captains = sorted(c for c in kept if re.fullmatch(rf"{name}/d\d+", c))
    assert len(captains) == 2 and set(found["agents"]) == {name, *captains}
    assert found["agents"][name]["agent"] == "lead"
    assert all(found["agents"][c]["agent"] == "captain" for c in captains)
    for key in (name, *captains):
        entry = found["agents"][key]
        assert entry["prompt"] == prompts(kept[key]) and entry["calls"] == len(entry["prompt"])
        assert entry["calls"] > 0 and entry["wall_s"] > 0
        assert entry["moves"] == await Store(pool).moves("moon", key)
    assert sorted(r.body["calls"] for r in records if r.agent in ("lead", "captain")) == sorted(
        entry["calls"] for entry in found["agents"].values()
    )
    [lead] = [r for r in records if r.agent == "lead"]
    assert found["agents"][name]["wall_s"] == pytest.approx(lead.duration_ms / 1000, abs=1.0)
    assert found["agents"][name]["wall_s"] <= found["running_s"] <= found["elapsed_s"]

    jobs = sorted(
        (step["at"], len(step["result"].encode()))
        for r in records
        for step in r.body["steps"]
        if step["tool"] == "Jobs" and not step["is_error"]
    )
    assert jobs and found["jobs"] == {
        "calls": len(jobs),
        "bytes": [size for _, size in jobs],
        "max": max(size for _, size in jobs),
        "total": sum(size for _, size in jobs),
    }

    moves = await pool.fetch(
        "select moves from windows where scope = 'moon'"
        " and (conversation = $1 or starts_with(conversation, $1 || '/'))",
        name,
    )
    assert found["moves"] == sum(row["moves"] for row in moves) > 0

    waits = await pool.fetchrow(
        "select count(*) as n, coalesce(sum(wait), 0) as waited from provider_log"
        " where scope = 'moon' and survey = $1 and status = 429",
        name,
    )
    assert waits["n"] > 0 and found["rate_limits"]["429s"] == waits["n"]
    assert found["rate_limits"]["waited_s"] == pytest.approx(waits["waited"], abs=1e-3)

    surveyors = [r.run_id for r in records if r.agent == "surveyor"]
    written = await pool.fetchval(
        "select count(*) from notes where scope = 'moon' and survey = $1"
        " and author = any($2::text[])",
        name,
        surveyors,
    )
    assert written == 10 and found["notes"] == {
        "surveyors": 10,
        "written": 10,
        "written_per_surveyor": {"min": 1, "mean": 1.0, "max": 1},
        "read_with_rereads": 0,
        "read_with_rereads_per_surveyor": {"min": 0, "mean": 0.0, "max": 0},
    }

    made = [m for messages in kept.values() for m in replies(messages)]
    assert found["models"] == {
        "fake": {
            "calls": len(made),
            "input": sum(m.usage.input for m in made),
            "cached": sum(m.usage.cache_read for m in made),
            "cache_creation": sum(m.usage.cache_creation for m in made),
            "output": sum(m.usage.output for m in made),
            "spend": pytest.approx(sum(cost_of("fake", m.usage) for m in made)),
        }
    }
    total = await pool.fetchval(
        "select total from spend where scope = 'moon' and survey = $1", name
    )
    assert found["spend"] == pytest.approx(total)
    assert 0 < found["models"]["fake"]["spend"] < total
    gated = await pool.fetchval(
        "select count(*) from gate_log where scope = 'moon' and survey = $1", name
    )
    assert found["gate"] == {"calls": gated, "refused": 0}
    assert found["records"] == dict(Counter(r.agent for r in records))
    assert found["lost"] == {"notices": 0, "agents": {}}
    checkers = [r.run_id for r in records if r.agent.startswith("crosscheck")]
    checked = await pool.fetchval(
        "select count(distinct notes.region) from notes join landed using (scope, survey, region)"
        " where notes.scope = 'moon' and notes.survey = $1 and notes.author = any($2::text[])"
        " and notes.kind in ('agree', 'dispute')",
        name,
        checkers,
    )
    assert checked > 0
    assert found["dossiers"] == {"landed": 10, "naming_a_neighbour": 0, "crosschecked": checked}

    board = Board(pool, "moon", name, live=lambda _: False)
    rows = await board.landed()
    assert await board.land(rows[0]) is False
    cited, sourced = next(
        (row, value)
        for row in rows
        for value in row["dossier"]["values"].values()
        if "sources" in value
    )
    beside = moon.region(cited["region"]).neighbours[0]
    sourced["sources"].append({"source": f"the notes of {beside}", "mark": "cited"})
    await board.update_landed(cited["region"], dossier=cited["dossier"])
    surveyed = {row["region"] for row in rows}
    noted, quoting = [
        row
        for row in rows
        if row is not cited and set(moon.region(row["region"]).neighbours) & surveyed
    ][:2]
    noted["dossier"]["notes"] += f" {moon.region(noted['region']).neighbours[-1]} agrees."
    await board.update_landed(noted["region"], dossier=noted["dossier"])
    near = sorted(set(moon.region(quoting["region"]).neighbours) & surveyed)[0]
    await board.note(near, "test", "finding", "ok")
    quoting["dossier"]["notes"] += " ok"
    await board.update_landed(quoting["region"], dossier=quoting["dossier"])
    crossed = next(row for row in rows if row["disputes"])
    await board.update_landed(crossed["region"], disputes=[])
    store = Store(pool)
    told = Message(
        role="user",
        content="[helper d9 · captain · lost]",
        notice=Notice(job="d9", agent="captain", run_id=None, kind="lost"),
    )
    await store.append("moon", name, await store.length("moon", name), [told])

    again = await measure.measure(pool, engine_pool, name)
    assert again["landed_twice"] == 1
    assert again["lost"] == {"notices": 1, "agents": {"captain": 1}}
    assert again["dossiers"] == {"landed": 10, "naming_a_neighbour": 2, "crosschecked": checked - 1}


async def test_measure_reads_the_boss_runs_of_the_app_shape(service, pool, engine_pool):
    await service()
    name = fresh()
    await harness.run(name, mode="fake", shape="app", size=4, cap=2.0)
    assert (await ended(name))["status"] == "done"
    found = await measure.measure(pool, engine_pool, name)
    kept = await stored(pool, name)
    records = await records_of(engine_pool, name)
    bosses = sorted(c for c in kept if c.startswith(f"{name}/boss/"))
    assert bosses and set(found["agents"]) == set(bosses)
    for key in bosses:
        entry = found["agents"][key]
        assert entry["agent"] == "boss" and entry["prompt"] == prompts(kept[key])
    assert sorted(r.body["calls"] for r in records if r.agent == "boss") == sorted(
        entry["calls"] for entry in found["agents"].values()
    )
    assert found["records"] == {"boss": len(bosses), "surveyor": 4}
    assert found["notes"]["surveyors"] == 4 and found["notes"]["written"] == 4
    assert found["jobs"] == {"calls": 0, "bytes": [], "max": 0, "total": 0}
    with pytest.raises(measure.MeasureError, match="measure: survey 'nobody' does not exist"):
        await measure.measure(pool, engine_pool, "nobody")


class Held(FakeGate):
    def __init__(self, tools, regions=None):
        super().__init__()
        self.tools = tools
        self.regions = regions
        self.waiting = Counter()

    async def decide(self, state, questions):
        tool = state.get("tool")
        if tool in self.tools and (self.regions is None or state.get("region") in self.regions):
            self.waiting[tool] += 1
            try:
                await self.tools[tool].wait()
            finally:
                self.waiting[tool] -= 1
        return await super().decide(state, questions)


async def until(check, seconds=ENDED):
    async with asyncio.timeout(seconds):
        while not await check():
            await asyncio.sleep(0.05)


async def test_measure_counts_the_notes_each_surveyor_reads(service, pool, engine_pool):
    planted, read = asyncio.Event(), asyncio.Event()
    gate = Held({"notes": planted, "Shell": read})
    await service(gate=gate)
    name = fresh()
    walk = [r for share in moon.shares(25) for r in share][:4]
    await harness.run(name, mode="fake", shape="app", size=4, cap=2.0)
    for region in walk:
        await write_note(pool, "moon", name, region, "test", "finding", f"planted on {region}")
    planted.set()

    async def all_held():
        return gate.waiting["Shell"] == len(walk)

    await until(all_held)
    read.set()
    assert (await ended(name))["status"] == "done"
    each = [len(set(moon.region(region).neighbours) & set(walk)) for region in walk]
    assert sum(each) > 0
    found = await measure.measure(pool, engine_pool, name)
    assert found["notes"] == {
        "surveyors": 4,
        "written": 4,
        "written_per_surveyor": {"min": 1, "mean": 1.0, "max": 1},
        "read_with_rereads": sum(each),
        "read_with_rereads_per_surveyor": {
            "min": min(each),
            "mean": round(sum(each) / len(each), 2),
            "max": max(each),
        },
    }


GAP = 1.5


async def test_a_crashed_surveys_numbers_leave_out_its_downtime(service, pool, engine_pool):
    walk = [r for share in moon.shares(25) for r in share][:10]
    released = asyncio.Event()
    gate = Held({"Shell": released}, regions=set(walk[5:]))
    await service(gate=gate, size=1)
    name = fresh()
    await harness.run(name, mode="fake", shape="crews", size=10, cap=2.0)

    async def half():
        counts = (await harness.progress(name))["counts"]
        return counts["landed"] == 5 and counts["claimed"] == 5

    await until(half)
    task = harness.held.surveys[name].task
    task.cancel()
    await asyncio.wait([task])
    await asyncio.sleep(GAP)
    released.set()
    await harness.resume(name)
    assert (await ended(name))["status"] == "done"

    found = await measure.measure(pool, engine_pool, name)
    records = await records_of(engine_pool, name)
    kept = await stored(pool, name)
    lead = found["agents"][name]
    [resumed] = [r for r in records if r.agent == "lead"]
    assert found["lost"]["notices"] > 0 and found["elapsed_s"] > GAP
    assert resumed.duration_ms / 1000 - 0.5 <= lead["wall_s"] <= found["elapsed_s"] - GAP
    assert lead["wall_s"] <= found["running_s"] <= found["elapsed_s"] - GAP

    recorded = Counter(r.agent for r in records)
    assert recorded["surveyor"] == 10 and found["notes"]["surveyors"] == 15
    answers = []
    for messages in kept.values():
        latest = {}
        for m in messages:
            latest.update({call.id: call.name for call in m.tool_calls})
            if m.role == "tool" and latest.pop(m.tool_call_id, None) == "Jobs":
                answers.append(m)
    in_records = [step for r in records for step in r.body["steps"] if step["tool"] == "Jobs"]
    assert found["jobs"]["calls"] == len(answers) > len(in_records)


def test_results_append_a_labelled_section(tmp_path):
    path = tmp_path / "results.md"
    path.write_text("# The Moon survey's results\n")
    rows = {
        "survey": "t-1",
        "label": "fake",
        "usage": "characters / 4, the fakes' estimate",
        "wall_s": 12.5,
        "said": "a | b",
        "agents": {
            "t-1": {"agent": "lead", "calls": 2, "prompt": [120, 340]},
            "t-1/d1": {"agent": "captain", "calls": 1, "prompt": [90]},
        },
        "lost": {"notices": 0, "agents": {}},
        "jobs": {"calls": 0, "bytes": []},
    }
    measure.append_results(path, "Fake crews of 10", rows)
    measure.append_results(
        path, "The full catalogue", {"label": "arithmetic", "dossiers": 540, "dollars": 12.25}
    )
    assert path.read_text() == "\n".join(
        [
            "# The Moon survey's results",
            "",
            "## Fake crews of 10",
            "",
            "`fake`",
            "",
            "| measure | value |",
            "|---|---|",
            "| survey | t-1 |",
            "| usage | characters / 4, the fakes' estimate |",
            "| wall_s | 12.5 |",
            "| said | a \\| b |",
            "| lost.notices | 0 |",
            "| lost.agents | none |",
            "| jobs.calls | 0 |",
            "| jobs.bytes | none |",
            "",
            "| agents | agent | calls | prompt |",
            "|---|---|---|---|",
            "| t-1 | lead | 2 | 120, 340 |",
            "| t-1/d1 | captain | 1 | 90 |",
            "",
            "## The full catalogue",
            "",
            "`arithmetic`",
            "",
            "| measure | value |",
            "|---|---|",
            "| dossiers | 540 |",
            "| dollars | 12.25 |",
            "",
        ]
    )
    before = path.read_text()
    for refused in ({}, {"label": "estimated"}):
        with pytest.raises(
            measure.MeasureError, match="measure: a section's label is real, fake or arithmetic"
        ):
            measure.append_results(path, "Unlabelled", refused)
    assert path.read_text() == before


def test_the_command_says_how_it_is_used(capsys, monkeypatch):
    assert measure.main([]) == 2
    assert "usage: uv run src/measure.py <survey> <results.md> [<title>]" in capsys.readouterr().err
    monkeypatch.delenv("RFNRY_EXAMPLES_POSTGRES_URL", raising=False)
    assert measure.main(["t-1", "results.md"]) == 1
    assert "measure: RFNRY_EXAMPLES_POSTGRES_URL is not set" in capsys.readouterr().err

import asyncio
import re
import secrets
import subprocess
import sys
from collections import Counter
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from rfnry_engine import PostgresStorage

import harness
import merge
import moon
import routes
from board import Board
from fakes import FakeGate
from merge import Price
from store import Store

EXAMPLE = Path(__file__).resolve().parent.parent
CUT = EXAMPLE / "tests" / "data" / "lp_cut"
KEYS = ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "TYPESAFE_API_KEY")
PRICE = Price(10.0, 10.0, 10.0, 10.0, 100_000)
CAP = 2.0
ENDED = 120


def fresh():
    return f"t-{secrets.token_hex(4)}"


@pytest_asyncio.fixture
async def service(pool, engine_pool, tmp_path, monkeypatch):
    monkeypatch.setenv("MOON_SURVEY_DATA", str(CUT))
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)

    async def opening(**given):
        settings = harness.Settings(work=tmp_path / "work", scale=0.0, per_minute=None, **given)
        await harness.open(pool, engine_pool, settings)

    yield opening
    await harness.close()


async def ended(name):
    return await asyncio.wait_for(harness.wait(name), ENDED)


def board_of(pool, name):
    return Board(pool, "moon", name, live=lambda _: False)


async def records_of(engine_pool, name):
    return await PostgresStorage(engine_pool).run_tree("moon", tags={"survey": name})


async def conversations_of(pool, name):
    rows = await pool.fetch(
        "select distinct conversation from messages"
        " where scope = 'moon' and (conversation = $1 or conversation like $1 || '/%')",
        name,
    )
    return {row["conversation"] for row in rows}


async def spent(pool, name):
    return await pool.fetchval("select total from spend where scope = 'moon' and survey = $1", name)


def test_the_folder_checks():
    cli = Path(sys.executable).parent / "rfnry-engine"
    done = subprocess.run(
        [str(cli), "folder", "check", "agents"], cwd=EXAMPLE, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    listed = {line.split()[0] for line in done.stdout.splitlines() if line.strip()}
    assert listed == {
        "lead",
        "flat",
        "captain",
        "surveyor",
        "crosscheck",
        "crosscheck_second",
        "boss",
        "gate",
    }


async def test_a_fake_survey_of_ten_lands_every_region(service, pool, engine_pool):
    await service()
    name = fresh()
    started = await harness.run(name, mode="fake", shape="crews", size=10, cap=2.0)
    assert started["status"] == "running"
    done = await ended(name)
    assert done["status"] == "ended" and done["error"] is None
    assert done["counts"] == {"landed": 10, "failed": 0, "claimed": 0, "free": 0}

    rows = await board_of(pool, name).landed()
    assert len(rows) == 10
    assert all(row["simulated"] and row["provider"] == "fake" for row in rows)
    assert all(row["attempts"] == 1 for row in rows)
    assert all(any(d["kind"] == "agree" for d in row["disputes"]) for row in rows)

    records = await records_of(engine_pool, name)
    assert Counter(r.agent for r in records) == {
        "lead": 1,
        "captain": 2,
        "surveyor": 10,
        "crosscheck_second": 10,
    }
    assert all(r.body["tags"] == {"survey": name} for r in records)
    steps = [step for r in records for step in r.body["steps"]]
    assert steps and all(step["approved"] is True for step in steps)
    gated = await pool.fetch(
        "select agent, tool, allowed, probability, error from gate_log"
        " where scope = 'moon' and survey = $1",
        name,
    )
    assert len(gated) == len(steps)
    assert all(g["allowed"] and g["probability"] == 1.0 and g["error"] is None for g in gated)
    assert Counter(g["tool"] for g in gated) == Counter(step["tool"] for step in steps)

    kept = await conversations_of(pool, name)
    assert len(kept) == len(records)
    below = sorted(c for c in kept if c != name)
    assert name in kept and all(re.fullmatch(rf"{name}(/d\d+)+", c) for c in below)
    assert Counter(c.count("/") for c in below) == {1: 2, 2: 20}


async def test_a_gate_that_cannot_answer_ends_the_survey(service, pool, engine_pool):
    await service(gate=FakeGate(down=True))
    name = fresh()
    await harness.run(name, mode="fake", shape="crews", size=10, cap=2.0)
    done = await ended(name)
    assert done["status"] == "ended" and done["error"] is None
    assert done["counts"] == {"landed": 0, "failed": 10, "claimed": 0, "free": 0}
    reasons = await pool.fetch(
        "select distinct reason from failures where scope = 'moon' and survey = $1", name
    )
    assert [r["reason"] for r in reasons] == ["the survey ended without it"]

    gated = await pool.fetch(
        "select allowed, probability, error from gate_log where scope = 'moon' and survey = $1",
        name,
    )
    assert gated and all(g["allowed"] is False and g["probability"] is None for g in gated)
    assert all("the fake gate is down" in g["error"] for g in gated)

    [lead] = await records_of(engine_pool, name)
    assert lead.agent == "lead" and lead.body["error"] is None
    assert lead.body["calls"] <= 4
    assert all(step["approved"] is False for step in lead.body["steps"])


async def test_the_cap_stops_the_survey_and_a_resume_goes_on(service, pool, monkeypatch):
    monkeypatch.setitem(merge.PRICES, "fake", PRICE)
    await service()
    name = fresh()
    await harness.run(name, mode="fake", shape="crews", size=10, cap=CAP)
    stopped = await ended(name)
    assert stopped["status"] == "stopped" and stopped["error"] is None
    assert stopped["counts"]["landed"] < 10
    assert stopped["spend"] >= CAP and stopped["spend"] == await spent(pool, name)
    assert stopped["cap"] == CAP

    with pytest.raises(harness.SurveyError, match="survey: "):
        await harness.run(name, mode="fake", shape="crews", size=10, cap=CAP)

    resumed = await harness.resume(name, cap=100.0)
    assert resumed["status"] == "running" and resumed["cap"] == 100.0
    done = await ended(name)
    assert done["status"] == "ended"
    assert done["counts"] == {"landed": 10, "failed": 0, "claimed": 0, "free": 0}
    assert done["spend"] > stopped["spend"] and done["spend"] == await spent(pool, name)


class Holding(FakeGate):
    def __init__(self, regions):
        super().__init__()
        self.regions = set(regions)
        self.released = asyncio.Event()

    async def decide(self, state, questions):
        if state.get("tool") == "Shell" and state.get("region") in self.regions:
            await self.released.wait()
        return await super().decide(state, questions)


async def until(check, seconds=ENDED):
    async with asyncio.timeout(seconds):
        while not await check():
            await asyncio.sleep(0.05)


async def test_a_crashed_survey_resumes(service, pool):
    walk = [r for share in moon.shares(25) for r in share][:10]
    gate = Holding(walk[5:])
    await service(gate=gate)
    name = fresh()
    await harness.run(name, mode="fake", shape="crews", size=10, cap=2.0)

    async def half():
        counts = (await harness.progress(name))["counts"]
        return counts["landed"] == 5 and counts["claimed"] == 5

    await until(half)
    held = {
        row["region"]: row["run_id"]
        for row in await pool.fetch(
            "select region, run_id from claims where scope = 'moon' and survey = $1", name
        )
        if row["region"] in walk[5:]
    }
    assert set(held) == set(walk[5:])
    task = harness.held.surveys[name].task
    task.cancel()
    await asyncio.wait([task])
    cut = await harness.progress(name)
    assert cut["status"] == "cancelled"
    assert cut["counts"] == {"landed": 5, "failed": 0, "claimed": 0, "free": 5}
    gate.released.set()

    await harness.resume(name)
    done = await ended(name)
    assert done["status"] == "ended"
    assert done["counts"] == {"landed": 10, "failed": 0, "claimed": 0, "free": 0}
    rows = {row["region"]: row for row in await board_of(pool, name).landed()}
    assert set(rows) == set(walk) and all(row["attempts"] == 1 for row in rows.values())
    claims = {
        row["region"]: row["run_id"]
        for row in await pool.fetch(
            "select region, run_id from claims where scope = 'moon' and survey = $1", name
        )
    }
    for region, run_id in held.items():
        assert claims[region] != run_id and claims[region] == rows[region]["run_id"]

    messages = await Store(pool).read("moon", name, 0)
    lost = [m for m in messages if m.notice is not None and m.notice.kind == "lost"]
    assert lost and all(m.notice.agent == "captain" for m in lost)
    inputs = [n for n, m in enumerate(messages) if m.role == "user" and m.notice is None]
    assert len(inputs) == 2
    assert all(messages.index(m) < inputs[1] for m in lost)
    assert "the process restarted" in messages[inputs[1]].content


async def test_hostile_mode_plants_its_note(service, pool):
    await service()
    name = fresh()
    walk = [r for share in moon.shares(25) for r in share]
    await harness.run(name, mode="hostile", shape="crews", size=2, cap=2.0, crews=1)
    notes = await board_of(pool, name).notes(walk[0])
    assert [(n["author"], n["kind"], n["text"]) for n in notes] == [
        ("planted", "finding", harness.PLANTED)
    ]
    assert "100% water ice" in harness.PLANTED and "stop the survey" in harness.PLANTED
    done = await ended(name)
    assert done["status"] == "failed" and "ANTHROPIC_API_KEY is not set" in done["error"]
    assert done["mode"] == "hostile"


async def test_the_routes_start_read_and_resume_a_survey(service, monkeypatch):
    monkeypatch.setitem(merge.PRICES, "fake", PRICE)
    await service()
    app = FastAPI()
    app.include_router(routes.router)
    name = fresh()
    body = {"name": name, "mode": "fake", "shape": "crews", "size": 10, "cap": CAP}

    async def read(client):
        async def settled():
            found = await client.get(f"/surveys/{name}")
            return found.json()["status"] != "running"

        await until(settled)
        return (await client.get(f"/surveys/{name}")).json()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://moon") as client:
        refused = await client.post("/surveys", json={**body, "shape": "app"})
        assert refused.status_code == 400
        assert refused.json()["detail"] == "survey: shape 'app' comes with the boss"
        assert (await client.get("/surveys/nobody")).status_code == 404

        started = await client.post("/surveys", json=body)
        assert started.status_code == 202 and started.json()["status"] == "running"
        assert (await client.post("/surveys", json=body)).status_code == 409
        stopped = await read(client)
        assert stopped["status"] == "stopped"

        resumed = await client.post(f"/surveys/{name}/resume", json={"cap": 100.0})
        assert resumed.status_code == 202 and resumed.json()["cap"] == 100.0
        done = await read(client)
        assert done["status"] == "ended" and done["counts"]["landed"] == 10

        shown = await client.get("/map", params={"survey": name})
        assert shown.status_code == 200
        regions = shown.json()["regions"]
        assert len(regions) == 10 and all(r["provider"] == "fake" for r in regions)
        assert all(r["dossier"] is None for r in regions)
        assert shown.json()["survey"] == name


def test_shares_are_even_contiguous_runs_and_crews_follow_the_size():
    walk = [r for share in moon.shares(25) for r in share]
    assert [len(s) for s in harness.split(walk[:10], harness.crews_of(10, None))] == [5, 5]
    assert [len(s) for s in harness.split(walk, harness.crews_of(250, None))] == [25] * 10
    assert [len(s) for s in harness.split(walk[:11], 3)] == [4, 4, 3]
    assert sum(harness.split(walk[:11], 3), []) == walk[:11]
    assert harness.crews_of(1, None) == 1 and harness.crews_of(10, 1) == 1
    with pytest.raises(harness.SurveyError, match="survey: crews must be at least 1"):
        harness.crews_of(10, 0)


def test_the_second_vendor_takes_the_truth_regions_the_north_cap_and_twenty_drawn():
    walk = [r for share in moon.shares(25) for r in share]
    truth = [{"region": r} for r in ("N00-15/18", "S00-15/14", "N30-45/9", "S75-90/0")]
    second = harness.second_of("survey-a", walk, truth)
    assert len(second) == 25 and {"N75-90/0", "N00-15/18", "S75-90/0"} <= second
    assert second == harness.second_of("survey-a", list(walk), truth)
    assert second != harness.second_of("survey-b", walk, truth)
    assert harness.second_of("survey-a", walk[:10], truth) == set(walk[:10])


async def test_the_harness_refuses_to_open_without_bubblewrap(pool, engine_pool, monkeypatch):
    monkeypatch.setenv("MOON_SURVEY_DATA", str(CUT))
    monkeypatch.setattr("sandbox.shutil.which", lambda *_, **__: None)
    with pytest.raises(harness.SurveyError, match="survey: bwrap was not found"):
        await harness.open(pool, engine_pool)
    assert harness.held is None

import asyncio
import json
import logging
import re
import secrets
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from rfnry_engine import PostgresStorage, RunEnded, RunStarted

import board as board_module
import harness
import merge
import moon
import providers
import routes
from board import Board, create_survey
from fakes import FakeGate
from merge import Price, cost_of
from providers import ProviderError
from store import Store

EXAMPLE = Path(__file__).resolve().parent.parent
PRICE = Price(10.0, 10.0, 10.0, 10.0, 100_000)
CAP = 2.0
APP_CAP = 0.5
ENDED = 120


def fresh():
    return f"t-{secrets.token_hex(4)}"


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
    assert done["status"] == "done" and done["error"] is None
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
    [lead] = [r for r in records if r.agent == "lead"]
    planned = json.loads(lead.body["context"])
    assert planned["name"] == name and set(planned["regions"]) == {row["region"] for row in rows}
    assert [len(share) for share in planned["shares"]] == [5, 5] and planned["check"] == []
    looked = [
        json.loads(step["result"])
        for r in records
        if r.agent.startswith("crosscheck")
        for step in r.body["steps"]
        if step["tool"] == "dossier"
    ]
    assert len(looked) == 10 and all(found["landed"] is not None for found in looked)
    assert {found["region"] for found in looked} == {row["region"] for row in rows}
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
    assert done["status"] == "done" and done["error"] is None
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

    with pytest.raises(harness.SurveyError, match=f"survey: survey '{name}' exists; resume it"):
        await harness.run(name, mode="fake", shape="crews", size=10, cap=CAP)
    with pytest.raises(harness.SurveyError) as at_cap:
        await harness.resume(name)
    assert at_cap.value.status == 409
    assert str(at_cap.value) == (
        f"survey: '{name}' stopped at its cap of $2; resume with a higher cap"
    )
    with pytest.raises(harness.SurveyError, match="at or below the .* so far") as below:
        await harness.resume(name, cap=stopped["spend"])
    assert below.value.status == 409

    await harness.close()
    await service()
    assert (await harness.progress(name))["status"] == "stopped"
    resumed = await harness.resume(name, cap=100.0)
    assert resumed["status"] == "running" and resumed["cap"] == 100.0
    done = await ended(name)
    assert done["status"] == "done"
    assert done["counts"] == {"landed": 10, "failed": 0, "claimed": 0, "free": 0}
    assert done["spend"] > stopped["spend"] and done["spend"] == await spent(pool, name)


class Holding(FakeGate):
    def __init__(self, regions, agents=()):
        super().__init__()
        self.regions = set(regions)
        self.agents = set(agents)
        self.released = asyncio.Event()
        self.waiting = 0

    async def decide(self, state, questions):
        shell = state.get("tool") == "Shell" and state.get("region") in self.regions
        if shell or state.get("agent") in self.agents:
            self.waiting += 1
            try:
                await self.released.wait()
            finally:
                self.waiting -= 1
        return await super().decide(state, questions)


async def until(check, seconds=ENDED):
    async with asyncio.timeout(seconds):
        while not await check():
            await asyncio.sleep(0.05)


async def test_a_crashed_survey_resumes(service, pool, engine_pool):
    walk = [r for share in moon.shares(25) for r in share][:10]
    gate = Holding(walk[5:], agents={"crosscheck", "crosscheck_second"})
    await service(gate=gate)
    name = fresh()
    await harness.run(name, mode="fake", shape="crews", size=10, cap=2.0)

    async def half():
        counts = (await harness.progress(name))["counts"]
        return counts["landed"] == 5 and counts["claimed"] == 5 and gate.waiting == 10

    await until(half)
    unchecked = await board_of(pool, name).landed()
    assert {row["region"] for row in unchecked} == set(walk[:5])
    assert all(row["disputes"] == [] for row in unchecked)
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
    assert cut["status"] == "interrupted"
    assert cut["counts"] == {"landed": 5, "failed": 0, "claimed": 0, "free": 5}
    gate.released.set()

    await harness.resume(name)
    done = await ended(name)
    assert done["status"] == "done"
    assert done["counts"] == {"landed": 10, "failed": 0, "claimed": 0, "free": 0}
    rows = {row["region"]: row for row in await board_of(pool, name).landed()}
    assert set(rows) == set(walk) and all(row["attempts"] == 1 for row in rows.values())
    assert all(any(d["kind"] == "agree" for d in row["disputes"]) for row in rows.values())
    records = await records_of(engine_pool, name)
    [resumed] = [r for r in records if r.agent == "lead"]
    planned = json.loads(resumed.body["context"])
    assert planned["regions"] == walk and planned["check"] == walk[:5]
    assert Counter(r.agent for r in records) == {
        "lead": 1,
        "captain": 2,
        "surveyor": 10,
        "crosscheck_second": 10,
    }
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
    assert [messages[n].content for n in inputs] == [harness.START, harness.RESTARTED]


async def test_a_survey_landed_and_crosschecked_throughout_has_nothing_to_resume(service, pool):
    await service()
    walk = [r for share in moon.shares(25) for r in share][:2]
    agreed = [{"note": 1, "run_id": "r-c", "kind": "agree", "name": None, "text": "fine"}]
    columns = {"dossier": {}, "checks": {}, "data_check": {}, "truth": None}
    said = {"crews": "has landed and been crosschecked", "app": "has landed"}
    for shape, done in said.items():
        name = fresh()
        await create_survey(
            pool, "moon", name, 2.0, set(), walk, mode="fake", shape=shape, crews=1, status="done"
        )
        for region in walk:
            row = {"region": region, "run_id": f"r-{region}", "provider": "fake", "simulated": True}
            disputes = agreed if shape == "crews" else []
            await board_of(pool, name).land({**row, **columns, "disputes": disputes})
        with pytest.raises(harness.SurveyError) as refused:
            await harness.resume(name)
        assert refused.value.status == 409
        assert str(refused.value) == (
            f"survey: every region of survey '{name}' {done}; there is nothing to resume"
        )


def offline_providers(monkeypatch):
    for key in providers.KEYS:
        monkeypatch.setenv(key, "not-a-key")

    def unreachable(model, effort):
        raise ProviderError("providers: this test reaches no model")

    for name in list(harness.PROVIDERS):
        monkeypatch.setitem(harness.PROVIDERS, name, unreachable)


async def test_hostile_mode_plants_its_note(service, pool, caplog, monkeypatch):
    caplog.set_level(logging.ERROR, logger="harness")
    offline_providers(monkeypatch)
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
    assert done["status"] == "failed"
    assert done["error"] == "ProviderError: providers: this test reaches no model"
    assert done["mode"] == "hostile"
    [logged] = [r for r in caplog.records if r.getMessage() == f"survey: survey '{name}' failed"]
    assert logged.exc_info is not None and "reaches no model" in str(logged.exc_info[1])
    await harness.close()
    await service()
    again = await harness.progress(name)
    assert again["status"] == "failed" and again["error"] == done["error"]


async def test_a_real_or_hostile_survey_needs_every_key_before_it_starts(
    service, pool, monkeypatch
):
    offline_providers(monkeypatch)
    monkeypatch.delenv("TYPESAFE_API_KEY")
    await service()
    for mode in ("real", "hostile"):
        name = fresh()
        with pytest.raises(harness.SurveyError) as refused:
            await harness.run(name, mode=mode, shape="crews", size=2, cap=2.0)
        assert refused.value.status == 503
        assert str(refused.value) == "survey: TYPESAFE_API_KEY is not set"
        with pytest.raises(harness.SurveyError, match="does not exist"):
            await harness.progress(name)
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(harness.SurveyError, match="^survey: ANTHROPIC_API_KEY is not set$"):
        await harness.run(fresh(), mode="real", shape="crews", size=2, cap=2.0)

    walk = [r for share in moon.shares(25) for r in share][:2]
    name = fresh()
    await create_survey(
        pool, "moon", name, 2.0, set(), walk, mode="real", shape="crews", crews=1, status="stopped"
    )
    for _ in range(2):
        with pytest.raises(harness.SurveyError) as refused:
            await harness.resume(name)
        assert refused.value.status == 503
        assert str(refused.value) == "survey: ANTHROPIC_API_KEY is not set"
    assert (await harness.progress(name))["status"] == "stopped"
    assert harness.held.surveys == {}
    await harness.run(fresh(), mode="fake", shape="crews", size=2, cap=2.0)


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
        invalid = [
            ({**body, "mode": "pretend"}, "survey: mode must be one of fake, real, hostile"),
            ({**body, "shape": "ring"}, "survey: shape must be one of crews, flat, app"),
            ({**body, "size": 0}, "survey: size must be from 1 to 250"),
            ({**body, "size": 251}, "survey: size must be from 1 to 250"),
            ({**body, "regions": ["N99-99/1"]}, "survey: these regions do not exist: N99-99/1"),
            (
                {**body, "regions": ["S75-90/0", "S75-90/0"]},
                "survey: these regions are named twice: S75-90/0",
            ),
            ({**body, "name": "a/b"}, "survey: a name is 1 to 64"),
            ({**body, "cap": "lots"}, "survey: cap must be a number of dollars"),
            ({**body, "crews": 11}, "survey: crews must be at most 10"),
            ({**body, "colour": "red"}, "survey: the body takes name, mode, shape"),
            (["not", "an", "object"], "survey: the body must be a JSON object"),
        ]
        for sent, said in invalid:
            refused = await client.post("/surveys", json=sent)
            assert refused.status_code == 400, sent
            assert refused.json()["detail"].startswith(said), refused.json()
        assert (await client.get("/surveys/nobody")).status_code == 404

        started = await client.post("/surveys", json=body)
        assert started.status_code == 202 and started.json()["status"] == "running"
        other = await client.post("/surveys", json={**body, "name": fresh()})
        assert other.status_code == 409
        assert other.json()["detail"] == f"survey: '{name}' is running"
        stopped = await read(client)
        assert stopped["status"] == "stopped"
        twice = await client.post("/surveys", json=body)
        assert twice.status_code == 409
        assert twice.json()["detail"] == f"survey: survey '{name}' exists; resume it"
        at_cap = await client.post(f"/surveys/{name}/resume")
        assert at_cap.status_code == 409 and "stopped at its cap" in at_cap.json()["detail"]

        resumed = await client.post(f"/surveys/{name}/resume", json={"cap": 100.0})
        assert resumed.status_code == 202 and resumed.json()["cap"] == 100.0
        done = await read(client)
        assert done["status"] == "done" and done["counts"]["landed"] == 10

        shown = await client.get("/map", params={"survey": name})
        assert shown.status_code == 200
        regions = shown.json()["regions"]
        assert len(regions) == 10 and all(r["provider"] == "fake" for r in regions)
        assert all(r["dossier"] is None for r in regions)
        assert shown.json()["survey"] == name


async def test_a_window_that_moves_mid_survey_still_lands_and_crosschecks_every_region_once(
    service, pool, engine_pool
):
    await service(size=1)
    name = fresh()
    await harness.run(name, mode="fake", shape="crews", size=10, cap=2.0)
    done = await ended(name)
    assert done["status"] == "done" and done["error"] is None
    assert done["counts"] == {"landed": 10, "failed": 0, "claimed": 0, "free": 0}
    rows = await board_of(pool, name).landed()
    assert len(rows) == 10 and all(row["attempts"] == 1 for row in rows)
    assert all(any(d["kind"] == "agree" for d in row["disputes"]) for row in rows)
    records = await records_of(engine_pool, name)
    assert Counter(r.agent for r in records) == {
        "lead": 1,
        "captain": 2,
        "surveyor": 10,
        "crosscheck_second": 10,
    }
    store = Store(pool)
    for conversation in (name, f"{name}/d1", f"{name}/d2"):
        assert await store.moves("moon", conversation) > 0, conversation
    messages = await store.read("moon", name, 0)
    assert any(call.name == "Jobs" for m in messages for call in m.tool_calls)
    for captain in (f"{name}/d1", f"{name}/d2"):
        held = await store.read("moon", captain, 0)
        assert {"History", "Jobs"} <= {call.name for m in held for call in m.tool_calls}


async def test_one_survey_runs_at_a_time(service):
    await service(gate=Holding(set(moon.by_id())))
    first, second = fresh(), fresh()
    raced = await asyncio.gather(
        *(harness.run(n, mode="fake", shape="crews", size=4, cap=2.0) for n in (first, second)),
        return_exceptions=True,
    )
    assert isinstance(raced[0], dict) and isinstance(raced[1], harness.SurveyError)
    assert str(raced[1]) == f"survey: '{first}' is running"
    with pytest.raises(harness.SurveyError, match=f"survey: '{first}' is running") as refused:
        await harness.run(second, mode="fake", shape="crews", size=4, cap=2.0)
    assert refused.value.status == 409
    with pytest.raises(harness.SurveyError, match=f"survey: '{first}' is running"):
        await harness.resume(first)
    task = harness.held.surveys[first].task
    task.cancel()
    await asyncio.wait([task])
    await harness.run(second, mode="fake", shape="flat", size=1, cap=2.0)


async def test_crews_and_shares_past_the_folders_max_background_are_refused(service):
    await service()
    walk = [r for share in moon.shares(25) for r in share]
    with pytest.raises(harness.SurveyError, match="survey: crews must be at most 10"):
        await harness.run(fresh(), mode="fake", shape="crews", size=20, cap=2.0, crews=11)
    with pytest.raises(harness.SurveyError, match="a share of 30 regions is over .* of 25"):
        await harness.run(fresh(), mode="fake", shape="crews", regions=walk[:30], cap=2.0, crews=1)
    assert harness.held.surveys == {}


async def test_a_gate_check_that_never_answers_is_refused_at_the_deadline(
    service, pool, monkeypatch
):
    async def hangs(self, run_id):
        await asyncio.Event().wait()

    monkeypatch.setattr(harness, "GATE_TIMEOUT", 0.2)
    monkeypatch.setattr(board_module.Board, "claim_of", hangs)
    await service()
    name = fresh()
    await harness.run(name, mode="fake", shape="crews", size=4, cap=2.0)
    done = await asyncio.wait_for(harness.wait(name), 20)
    assert done["status"] == "done"
    assert done["counts"] == {"landed": 0, "failed": 4, "claimed": 0, "free": 0}
    gated = await pool.fetch(
        "select allowed, probability, error from gate_log where scope = 'moon' and survey = $1",
        name,
    )
    assert gated and all(
        (g["allowed"], g["probability"], g["error"]) == (False, None, "timeout") for g in gated
    )


async def test_a_region_the_merge_failed_is_not_failed_again_as_the_survey_ends(service, pool):
    await service(gate=FakeGate(refuse=frozenset({"Write"})))
    name = fresh()
    await harness.run(name, mode="fake", shape="crews", size=4, cap=2.0)
    done = await ended(name)
    assert done["counts"] == {"landed": 0, "failed": 4, "claimed": 0, "free": 0}
    rows = await pool.fetch(
        "select region, reason from failures where scope = 'moon' and survey = $1", name
    )
    assert len(rows) == 4 and {r["reason"] for r in rows} == {"merge: no dossier.json"}


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


async def test_the_harness_refuses_to_open_without_bubblewrap(
    pool, engine_pool, offline, monkeypatch
):
    monkeypatch.setattr("sandbox.shutil.which", lambda *_, **__: None)
    with pytest.raises(harness.SurveyError, match="survey: bwrap was not found"):
        await harness.open(pool, engine_pool)
    assert harness.held is None


class Recording(FakeGate):
    def __init__(self):
        super().__init__()
        self.states = []

    async def decide(self, state, questions):
        self.states.append(state)
        return await super().decide(state, questions)


class Watching(FakeGate):
    def __init__(self):
        super().__init__()
        self.seen = []

    async def decide(self, state, questions):
        live = harness.held.surveys[harness.held.last].live
        self.seen.append((state, live.get(state["run_id"]), set(live)))
        return await super().decide(state, questions)


def first_sentence(text):
    return text.strip().splitlines()[0].split(". ")[0].rstrip(".")


async def test_the_gate_is_told_what_each_tool_does(service, engine_pool):
    gate = Watching()
    await service(gate=gate, size=1)
    name = fresh()
    await harness.run(name, mode="fake", shape="crews", size=10, cap=2.0)
    assert (await ended(name))["status"] == "done"
    records = {r.run_id: r for r in await records_of(engine_pool, name)}
    does = {}
    for state, began, _ in gate.seen:
        assert isinstance(began, RunStarted) and began.run_id == state["run_id"]
        [spec] = [tool for tool in began.tools if tool.name == state["tool"]]
        assert state["tool_does"] == first_sentence(spec.description)
        body = records[state["run_id"]].body
        offered = {tool["name"]: tool["description"] for tool in body["tools"]}
        assert offered[state["tool"]] == spec.description
        assert state["task"] == began.input == body["input"]
        does.setdefault((state["agent"], state["tool"]), set()).add(state["tool_does"])
    [jobs] = does[("lead", "Jobs")]
    assert jobs.startswith("List the helpers you started in the background")
    assert does[("captain", "Jobs")] == {jobs}
    [history] = does[("captain", "History")]
    assert history.startswith("Read the earlier messages of this conversation")
    assert does[("surveyor", "Shell")] == {"Run a shell command"}
    [claim] = does[("surveyor", "claim")]
    assert claim.startswith("Take a region for yourself")
    assert "\n" not in claim
    summed = [(s, live) for s, _, live in gate.seen if (s["agent"], s["tool"]) == ("lead", "note")]
    assert summed and all(live == {s["run_id"]} for s, live in summed)
    assert harness.held.surveys[name].live == {}


def test_a_calls_region_stands_to_the_reference_region():
    home = moon.regions()[0].id
    near = moon.region(home).neighbours[0]
    far = next(
        r.id for r in moon.regions() if r.id != home and r.id not in moon.region(home).neighbours
    )
    call = harness.call_region
    assert call(home, None, {"region": home}) == "its own region"
    assert call(home, None, {"region": near}) == "a neighbour of its region"
    assert call(home, None, {"region": far}) == "neither"
    assert call(None, None, {"region": near}) is None
    assert call(None, "not json", {"region": near}) is None
    assert call(home, None, {"command": "ls"}) is None
    task = json.dumps({"region": home})
    assert call(None, task, {"region": home}) == "its own region"
    assert call(None, task, {"region": near}) == "a neighbour of its region"
    assert call(None, task, {"region": far}) == "neither"


async def test_the_gate_is_told_how_a_calls_region_stands_to_the_callers(service):
    gate = Recording()
    await service(gate=gate)
    name = fresh()
    await harness.run(name, mode="fake", shape="crews", size=2, cap=2.0, crews=1)
    assert (await ended(name))["status"] == "done"
    assert all("call_region" in state for state in gate.states)
    read = [s for s in gate.states if s["agent"] == "surveyor" and s["tool"] == "notes"]
    assert read and {s["call_region"] for s in read} == {"a neighbour of its region"}
    claims = [s for s in gate.states if s["agent"] == "surveyor" and s["tool"] == "claim"]
    assert claims and all(s["call_region"] == "its own region" for s in claims)
    looked = [
        s for s in gate.states if s["agent"] == "crosscheck_second" and s["tool"] == "dossier"
    ]
    assert looked and all(s["region"] is None for s in looked)
    assert {s["call_region"] for s in looked} == {"its own region"}


async def test_the_gate_sees_each_callers_task(service):
    gate = Recording()
    await service(gate=gate)
    name = fresh()
    walk = [r for share in moon.shares(25) for r in share][:2]
    await harness.run(name, mode="fake", shape="crews", size=2, cap=2.0, crews=1)
    assert (await ended(name))["status"] == "done"
    assert gate.states and all(isinstance(state["task"], str) for state in gate.states)
    by = {state["agent"]: [] for state in gate.states}
    for state in gate.states:
        by[state["agent"]].append(state)
    assert {state["task"] for state in by["lead"]} == {harness.START}
    assert [json.loads(state["task"])["share"] for state in by["captain"]] == [walk] * len(
        by["captain"]
    )
    looked = [state for state in by["crosscheck_second"] if state["tool"] == "dossier"]
    assert len(looked) == 2 and all(state["region"] is None for state in looked)
    assert {state["task"] for state in looked} == {json.dumps({"region": r}) for r in walk}
    assert all(json.loads(state["task"]) == {"region": state["args"]["region"]} for state in looked)
    for state in by["surveyor"]:
        assert json.loads(state["task"])["region"] in walk


async def test_a_survey_whose_regions_cannot_be_failed_still_ends_failed(
    service, pool, monkeypatch
):
    said = iter(["board: the database went away", "board: the database is still away"])

    async def broken(board, run_id):
        raise RuntimeError(next(said))

    monkeypatch.setattr(harness, "fail_left", broken)
    await service(gate=FakeGate(down=True))
    name = fresh()
    await harness.run(name, mode="fake", shape="crews", size=2, cap=2.0)
    done = await ended(name)
    assert done["status"] == "failed"
    assert done["error"] == "RuntimeError: board: the database went away"
    assert next(said, None) is None
    stored = await pool.fetchrow(
        "select status, error from surveys where scope = 'moon' and survey = $1", name
    )
    assert (stored["status"], stored["error"]) == ("failed", done["error"])
    await harness.close()
    await service()
    again = await harness.progress(name)
    assert (again["status"], again["error"]) == ("failed", done["error"])


async def test_only_the_top_run_reads_the_survey(service, engine_pool):
    await service()
    for shape, size, top in (("crews", 4, "lead"), ("flat", 3, "flat")):
        name = fresh()
        await harness.run(name, mode="fake", shape=shape, size=size, cap=2.0)
        assert (await ended(name))["status"] == "done"
        records = await records_of(engine_pool, name)
        [lead] = [r for r in records if r.agent == top]
        assert json.loads(lead.body["context"])["name"] == name
        helpers = [r for r in records if r.agent != top]
        assert helpers and all(r.body["context"] is None for r in helpers)
        assert {r.agent for r in helpers} >= {"surveyor", "crosscheck_second"}


class Counting(FakeGate):
    def __init__(self):
        super().__init__()
        self.used = []

    async def decide(self, state, questions):
        decided = await super().decide(state, questions)
        self.used.append(decided.usage)
        return decided


async def test_every_model_call_is_priced_once(service, pool, monkeypatch):
    monkeypatch.setitem(merge.PRICES, "fake", PRICE)
    priced = []
    replied = merge.Spend.replied

    async def counted(self, event):
        priced.append(event)
        await replied(self, event)

    monkeypatch.setattr(merge.Spend, "replied", counted)
    gate = Counting()
    await service(gate=gate)
    store = Store(pool)
    for shape, size in (("crews", 4), ("flat", 3), ("app", 2)):
        priced.clear()
        gate.used.clear()
        name = fresh()
        await harness.run(name, mode="fake", shape=shape, size=size, cap=1000.0)
        assert (await ended(name))["status"] == "done"
        replies = [
            m
            for conversation in await conversations_of(pool, name)
            for m in await store.read("moon", conversation, 0)
            if m.role == "assistant"
        ]
        assert replies and gate.used and all(m.usage is not None for m in replies)
        kept = Counter((m.model, m.usage.input, m.usage.output) for m in replies)
        decided = Counter(("fake", u.input, u.output) for u in gate.used)
        assert Counter((e.model, e.usage.input, e.usage.output) for e in priced) == kept + decided
        total = sum(cost_of(m.model, m.usage) for m in replies)
        total += sum(cost_of("fake", u) for u in gate.used)
        assert await spent(pool, name) == pytest.approx(total)


async def test_the_crew_limits_are_the_engines_view_of_the_folder(service):
    await service()
    engine = harness.held.engine
    assert harness.background_of(engine, "lead") == 10 == harness.held.crews
    assert harness.background_of(engine, "captain") == 25 == harness.held.share
    assert harness.background_of(engine, "surveyor") is None


def boss_runs(records):
    return sorted((r for r in records if r.agent == "boss"), key=lambda r: r.at)


async def test_the_app_shape_lands_every_region(service, pool, engine_pool):
    await service()
    name = fresh()
    started = await harness.run(name, mode="fake", shape="app", size=10, cap=2.0)
    assert started["status"] == "running" and started["shape"] == "app"
    done = await ended(name)
    assert done["status"] == "done" and done["error"] is None
    assert done["counts"] == {"landed": 10, "failed": 0, "claimed": 0, "free": 0}
    rows = await board_of(pool, name).landed()
    assert len(rows) == 10 and all(row["attempts"] == 1 for row in rows)
    assert all(row["simulated"] and row["provider"] == "fake" for row in rows)

    records = await records_of(engine_pool, name)
    agents = Counter(r.agent for r in records)
    assert set(agents) == {"boss", "surveyor"} and agents["surveyor"] == 10
    assert all(r.parent_run_id is None and r.body["call"] is None for r in records)
    assert all(r.body["tags"] == {"survey": name} for r in records)
    bosses = boss_runs(records)
    assert bosses[0].body["input"] == harness.START
    assert all(boss.body["input"] == harness.NEXT for boss in bosses[1:])
    first = bosses[0].body["steps"]
    assert [step["tool"] for step in first] == ["progress"] + ["start"] * 10 + ["check"] * 10
    walk = [r for share in moon.shares(25) for r in share][:10]
    read = json.loads(first[0]["result"])
    assert read == {
        "counts": {"landed": 0, "failed": 0, "claimed": 0, "free": 10},
        "free": walk,
        "running": 0,
    }
    starts = [step for step in first if step["tool"] == "start"]
    assert [step["args"]["region"] for step in starts] == walk
    jobs = [json.loads(step["result"])["job"] for step in starts]
    assert sorted(jobs, key=lambda job: int(job[1:])) == [f"s{n}" for n in range(1, 11)]
    checked = [json.loads(step["result"]) for step in first if step["tool"] == "check"]
    assert [(found["job"], found["region"]) for found in checked] == list(
        zip(jobs, walk, strict=True)
    )
    assert all(boss.body["steps"][0]["tool"] == "progress" for boss in bosses)
    refused = [
        step
        for boss in bosses
        for step in boss.body["steps"]
        if step["tool"] == "start" and step["is_error"]
    ]
    assert refused == []
    surveyed = {json.loads(r.body["input"])["region"] for r in records if r.agent == "surveyor"}
    assert surveyed == {row["region"] for row in rows}

    kept = await conversations_of(pool, name)
    assert kept == {f"{name}/boss/{n}" for n in range(1, len(bosses) + 1)} | {
        f"{name}/s/{region}/1" for region in surveyed
    }
    steps = [step for r in records for step in r.body["steps"]]
    gated = await pool.fetch(
        "select tool, allowed from gate_log where scope = 'moon' and survey = $1", name
    )
    assert Counter(g["tool"] for g in gated) == Counter(step["tool"] for step in steps)
    assert all(g["allowed"] for g in gated)


async def test_the_app_shape_ends_when_its_gate_cannot_answer(service, pool, engine_pool):
    await service(gate=FakeGate(down=True))
    name = fresh()
    await harness.run(name, mode="fake", shape="app", size=4, cap=2.0)
    done = await ended(name)
    assert done["status"] == "done" and done["error"] is None
    assert done["counts"] == {"landed": 0, "failed": 4, "claimed": 0, "free": 0}
    [boss] = await records_of(engine_pool, name)
    assert boss.agent == "boss" and boss.body["error"] is None
    assert boss.body["steps"] and all(step["approved"] is False for step in boss.body["steps"])


async def test_the_app_shape_starts_a_region_at_most_twice(service, pool, engine_pool):
    await service(gate=FakeGate(refuse=frozenset({"Write"})))
    name = fresh()
    await harness.run(name, mode="fake", shape="app", size=3, cap=2.0)
    done = await ended(name)
    assert done["status"] == "done"
    assert done["counts"] == {"landed": 0, "failed": 3, "claimed": 0, "free": 0}
    records = await records_of(engine_pool, name)
    surveyed = Counter(
        json.loads(r.body["input"])["region"] for r in records if r.agent == "surveyor"
    )
    assert len(surveyed) == 3 and set(surveyed.values()) == {2}
    kept = await conversations_of(pool, name)
    assert {c for c in kept if "/s/" in c} == {
        f"{name}/s/{region}/{attempt}" for region in surveyed for attempt in (1, 2)
    }
    reasons = await pool.fetch(
        "select region, reason from failures where scope = 'moon' and survey = $1", name
    )
    assert Counter(r["region"] for r in reasons) == {region: 2 for region in surveyed}
    assert {r["reason"] for r in reasons} == {"merge: no dossier.json"}
    refused = [
        step
        for boss in boss_runs(records)
        for step in boss.body["steps"]
        if step["tool"] == "start" and step["is_error"]
    ]
    assert refused == []


async def test_the_app_shape_stops_at_its_cap_and_a_resume_lands_the_rest(
    service, pool, engine_pool, monkeypatch
):
    monkeypatch.setitem(merge.PRICES, "fake", PRICE)
    await service()
    name = fresh()
    await harness.run(name, mode="fake", shape="app", size=10, cap=APP_CAP)
    stopped = await ended(name)
    assert stopped["status"] == "stopped" and stopped["error"] is None
    assert stopped["counts"]["landed"] < 10 and stopped["counts"]["claimed"] == 0
    assert stopped["spend"] >= APP_CAP

    await harness.resume(name, cap=100.0)
    done = await ended(name)
    assert done["status"] == "done"
    assert done["counts"] == {"landed": 10, "failed": 0, "claimed": 0, "free": 0}
    rows = await board_of(pool, name).landed()
    assert all(row["attempts"] == 1 for row in rows)

    records = await records_of(engine_pool, name)
    bosses = boss_runs(records)
    inputs = [boss.body["input"] for boss in bosses]
    assert inputs[0] == harness.START and harness.WHY["stopped"] in inputs
    assert inputs.count(harness.START) == 1 and inputs.count(harness.WHY["stopped"]) == 1
    kept = await conversations_of(pool, name)
    numbers = sorted(int(c.rsplit("/", 1)[1]) for c in kept if c.startswith(f"{name}/boss/"))
    assert numbers == list(range(1, len(bosses) + 1))
    attempts = Counter(c.rsplit("/", 1)[0] for c in kept if c.startswith(f"{name}/s/"))
    for key, count in attempts.items():
        assert {f"{key}/{n}" for n in range(1, count + 1)} <= kept


async def test_the_boss_tools_refuse_what_cannot_start(pool):
    name = fresh()
    first, second = [r for share in moon.shares(25) for r in share][:2]
    await create_survey(
        pool,
        "moon",
        name,
        2.0,
        set(),
        [first, second],
        mode="fake",
        shape="app",
        crews=1,
        status="running",
    )
    keys, released = [], [asyncio.Event(), asyncio.Event()]

    async def run(agent, given, key, *, tools, ended):
        keys.append(key)
        await released[len(keys) - 1].wait()
        ended(RunEnded(run_id=f"r-{len(keys)}", agent=agent, call=None, stopped=None, error=None))
        return SimpleNamespace(reply=f"{json.loads(given)['region']}: done", stopped=None)

    spend = SimpleNamespace(pressed=False)
    shared = SimpleNamespace(
        survey=SimpleNamespace(name=name),
        board=board_of(pool, name),
        opened=SimpleNamespace(pool=pool),
        spend=spend,
        tools=None,
        run=run,
    )
    surveyors = harness.Surveyors(shared, {first: 1})
    assert (await surveyors.progress())["free"] == [first, second]
    assert await surveyors.start(first) == {"job": "s1"}
    with pytest.raises(
        harness.SurveyError, match=f"survey: region '{first}' has a surveyor running, job s1"
    ):
        await surveyors.start(first)
    with pytest.raises(
        harness.SurveyError, match=f"survey: region 'N75-90/0' is not in survey '{name}'"
    ):
        await surveyors.start("N75-90/0")
    with pytest.raises(harness.SurveyError, match="survey: there is no job 's9'"):
        await surveyors.check("s9")
    assert await surveyors.check("s1") == {"job": "s1", "region": first, "running": True}
    assert await surveyors.progress() == {
        "counts": {"landed": 0, "failed": 0, "claimed": 0, "free": 2},
        "free": [second],
        "running": 1,
    }
    released[0].set()
    await asyncio.wait_for(surveyors.finished(), 10)
    assert await surveyors.check("s1") == {
        "job": "s1",
        "region": first,
        "running": False,
        "ended": "finished",
        "reply": f"{first}: done",
        "error": None,
        "landed": False,
    }
    assert await surveyors.start(first) == {"job": "s2"}
    released[1].set()
    await asyncio.wait_for(surveyors.finished(), 10)
    with pytest.raises(harness.SurveyError, match=f"survey: region '{first}' was started 2 times"):
        await surveyors.start(first)
    assert keys == [f"{name}/s/{first}/2", f"{name}/s/{first}/3"]

    landed = {"region": second, "run_id": "r-x", "provider": "fake", "simulated": True}
    columns = {"dossier": {}, "checks": {}, "data_check": {}, "truth": None, "disputes": []}
    await board_of(pool, name).land({**landed, **columns})
    with pytest.raises(harness.SurveyError, match=f"survey: region '{second}' has landed"):
        await surveyors.start(second)
    assert (await surveyors.progress())["free"] == [] and await surveyors.settled()
    spend.pressed = True
    with pytest.raises(
        harness.SurveyError, match="survey: the survey is stopping at its spending cap"
    ):
        await surveyors.start(second)


async def test_a_crashed_app_shape_survey_resumes(service, pool, engine_pool):
    walk = [r for share in moon.shares(25) for r in share][:10]
    gate = Holding(walk[5:])
    await service(gate=gate)
    name = fresh()
    await harness.run(name, mode="fake", shape="app", size=10, cap=2.0)

    async def half():
        counts = (await harness.progress(name))["counts"]
        return counts["landed"] == 5 and counts["claimed"] == 5 and gate.waiting == 5

    await until(half)
    task = harness.held.surveys[name].task
    task.cancel()
    await asyncio.wait([task])
    assert gate.waiting == 0
    cut = await harness.progress(name)
    assert cut["status"] == "interrupted"
    assert cut["counts"] == {"landed": 5, "failed": 0, "claimed": 0, "free": 5}
    before = await conversations_of(pool, name)
    gate.released.set()

    await harness.resume(name)
    done = await ended(name)
    assert done["status"] == "done"
    assert done["counts"] == {"landed": 10, "failed": 0, "claimed": 0, "free": 0}
    rows = await board_of(pool, name).landed()
    assert {row["region"] for row in rows} == set(walk)
    assert all(row["attempts"] == 1 for row in rows)

    kept = await conversations_of(pool, name)
    assert before < kept
    numbers = sorted(int(c.rsplit("/", 1)[1]) for c in kept if c.startswith(f"{name}/boss/"))
    assert numbers == list(range(1, len(numbers) + 1))
    first = max(int(c.rsplit("/", 1)[1]) for c in before if c.startswith(f"{name}/boss/"))
    assert numbers[-1] > first
    assert {c for c in kept if "/s/" in c} == {f"{name}/s/{r}/1" for r in walk} | {
        f"{name}/s/{r}/2" for r in walk[5:]
    }
    records = await records_of(engine_pool, name)
    inputs = [boss.body["input"] for boss in boss_runs(records)]
    assert inputs[0] == harness.START and inputs.count(harness.RESTARTED) == 1


async def test_a_region_that_lands_while_the_boss_reads_is_neither_offered_nor_started(pool):
    name = fresh()
    walk = [r for share in moon.shares(25) for r in share][:3]
    await create_survey(
        pool,
        "moon",
        name,
        2.0,
        set(),
        walk,
        mode="fake",
        shape="app",
        crews=1,
        status="running",
    )
    board = board_of(pool, name)
    released = {region: asyncio.Event() for region in walk}

    async def run(agent, given, key, *, tools, ended):
        region = json.loads(given)["region"]
        await released[region].wait()
        columns = {"dossier": {}, "checks": {}, "data_check": {}, "truth": None, "disputes": []}
        await board.land(
            {"region": region, "run_id": key, "provider": "fake", "simulated": True, **columns}
        )
        ended(RunEnded(run_id=key, agent=agent, call=None, stopped=None, error=None))
        return SimpleNamespace(reply=f"{region}: done", stopped=None)

    shared = SimpleNamespace(
        survey=SimpleNamespace(name=name),
        board=board,
        opened=SimpleNamespace(pool=pool),
        spend=SimpleNamespace(pressed=False),
        tools=None,
        run=run,
    )
    surveyors = harness.Surveyors(shared, {})
    read = surveyors.landed
    landing = []

    async def racing():
        found = await read()
        if landing:
            region = landing.pop()
            released[region].set()
            [job] = [job for job in surveyors.jobs.values() if job.region == region]
            await asyncio.wait_for(job.task, 10)
        return found

    surveyors.landed = racing
    await surveyors.start(walk[0])
    await surveyors.start(walk[1])
    landing.append(walk[0])
    assert (await surveyors.progress())["free"] == [walk[2]]
    landing.append(walk[1])
    with pytest.raises(harness.SurveyError, match=f"survey: region '{walk[1]}' has landed"):
        await surveyors.start(walk[1])

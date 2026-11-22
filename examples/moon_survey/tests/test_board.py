import asyncio
import secrets
from pathlib import Path

import pytest
from rfnry_engine import CallInfo

import board as board_module
import moon
from board import Board, BoardError, board_tools, create_survey, write_note

REGIONS = [r.id for r in moon.regions()]
CUT = Path(__file__).resolve().parent / "data" / "lp_cut"
SECOND = {REGIONS[3], REGIONS[4]}


@pytest.fixture
def scope():
    return f"t{secrets.token_hex(4)}"


class Flight:
    def __init__(self, *running):
        self.running = set(running)

    def __call__(self, run_id):
        return run_id in self.running


@pytest.fixture
def caller(monkeypatch):
    async def run(tools, name, args, run_id="a"):
        monkeypatch.setattr(
            board_module,
            "current_call",
            lambda: CallInfo(
                run_id=run_id, agent="surveyor", scope="moon", tool=name, call_id="c1"
            ),
        )
        return await tools.dispatch(name, args)

    return run


async def made(pool, scope, flight=None, regions=None, second=SECOND):
    flight = flight or Flight("a", "b", "c")
    await create_survey(
        pool,
        scope,
        "s",
        5.0,
        second,
        regions or REGIONS,
        mode="fake",
        shape="crews",
        crews=1,
        status="running",
    )
    return Board(pool, scope, "s", live=flight), flight


def row(region, run_id="a", **more):
    return {
        "region": region,
        "run_id": run_id,
        "provider": "fake",
        "simulated": True,
        "dossier": {"k": 1},
        "checks": {"ok": True},
        "data_check": {"ok": True},
        "truth": {"t": 1},
        "disputes": [],
        **more,
    }


async def test_a_free_region_is_claimed(pool, scope):
    board, _ = await made(pool, scope)
    assert await board.claim(REGIONS[0], "a") == {"claimed": True, "by": "a"}
    assert await board.claim_of("a") == REGIONS[0]
    assert await board.claim_of("b") is None


async def test_a_live_holder_keeps_its_claim(pool, scope):
    board, _ = await made(pool, scope)
    await board.claim(REGIONS[0], "a")
    assert await board.claim(REGIONS[0], "b") == {"claimed": False, "by": "a"}
    assert await board.claim(REGIONS[0], "a") == {"claimed": True, "by": "a"}


async def test_a_claim_of_a_run_no_longer_in_flight_is_taken_over(pool, scope):
    board, flight = await made(pool, scope)
    await board.claim(REGIONS[0], "a")
    flight.running.discard("a")
    assert await board.claim(REGIONS[0], "b") == {"claimed": True, "by": "b"}
    assert await board.claim_of("a") is None


async def test_a_landed_region_is_never_claimed_again(pool, scope):
    board, flight = await made(pool, scope)
    await board.claim(REGIONS[0], "a")
    assert await board.land(row(REGIONS[0])) is True
    flight.running.clear()
    assert (await board.claim(REGIONS[0], "b"))["claimed"] is False


async def test_a_second_landing_keeps_the_first_and_counts_the_attempt(pool, scope):
    board, _ = await made(pool, scope)
    assert await board.land(row(REGIONS[0], "a")) is True
    assert await board.land(row(REGIONS[0], "b", dossier={"k": 2})) is False
    kept = await board.landed(REGIONS[0])
    assert len(kept) == 1
    assert kept[0]["run_id"] == "a" and kept[0]["dossier"] == {"k": 1}
    assert kept[0]["attempts"] == 2
    assert kept[0]["simulated"] is True and kept[0]["disputes"] == []


async def test_two_live_claimers_racing_cannot_both_win(pool, scope):
    flight = Flight(*[f"r{i}" for i in range(20)])
    board, _ = await made(pool, scope, flight)
    won = await asyncio.gather(*[board.claim(REGIONS[0], f"r{i}") for i in range(20)])
    assert sum(w["claimed"] for w in won) == 1
    assert len({w["by"] for w in won}) == 1


async def test_racing_takeovers_of_a_dead_claim_have_one_winner(pool, scope):
    flight = Flight("dead")
    board, _ = await made(pool, scope, flight)
    await board.claim(REGIONS[0], "dead")
    flight.running = {f"r{i}" for i in range(20)}
    won = await asyncio.gather(*[board.claim(REGIONS[0], f"r{i}") for i in range(20)])
    assert sum(w["claimed"] for w in won) == 1


async def test_a_run_holds_one_region(pool, scope):
    board, _ = await made(pool, scope)
    await board.claim(REGIONS[0], "a")
    with pytest.raises(BoardError, match=f"board: run 'a' already holds region '{REGIONS[0]}'"):
        await board.claim(REGIONS[1], "a")


async def test_a_region_outside_the_survey_is_refused(pool, scope):
    board, _ = await made(pool, scope, regions=REGIONS[:10])
    with pytest.raises(BoardError, match=f"board: region '{REGIONS[20]}' is not in survey 's'"):
        await board.claim(REGIONS[20], "a")


async def test_a_failed_region_can_be_claimed_again_but_a_landed_one_never(pool, scope):
    board, _ = await made(pool, scope)
    await board.claim(REGIONS[0], "a")
    await board.fail(REGIONS[0], "a", "ran out")
    assert (await board.counts())["failed"] == 1
    assert (await board.claim(REGIONS[0], "b"))["claimed"] is True
    await board.land(row(REGIONS[0], "b"))
    assert (await board.counts())["failed"] == 0


async def test_notes_come_back_in_order_with_their_author_and_kind(pool, scope):
    board, _ = await made(pool, scope)
    first = await board.note(REGIONS[0], "a", "finding", "one")
    second = await board.note(REGIONS[0], "b", "summary", "two")
    await board.note(REGIONS[1], "a", "agree", "elsewhere")
    assert second > first
    got = await board.notes(REGIONS[0])
    assert [(n["author"], n["kind"], n["text"]) for n in got] == [
        ("a", "finding", "one"),
        ("b", "summary", "two"),
    ]
    assert [n["text"] for n in await board.notes_by("a")] == ["one", "elsewhere"]


async def test_an_unknown_kind_is_refused(pool, scope):
    board, _ = await made(pool, scope)
    with pytest.raises(
        BoardError, match="board: kind must be one of finding, summary, agree, dispute; got 'x'"
    ):
        await board.note(REGIONS[0], "a", "x", "t")


async def test_an_unknown_region_is_refused(pool, scope):
    board, _ = await made(pool, scope)
    with pytest.raises(BoardError, match="board: region 'X9' does not exist"):
        await board.claim("X9", "a")
    with pytest.raises(BoardError, match="board: region 'X9' does not exist"):
        await board.note("X9", "a", "finding", "t")


async def test_surveys_do_not_see_each_other(pool, scope):
    board, flight = await made(pool, scope)
    await create_survey(
        pool,
        scope,
        "other",
        1.0,
        set(),
        REGIONS,
        mode="fake",
        shape="crews",
        crews=1,
        status="running",
    )
    other = Board(pool, scope, "other", live=flight)
    await board.claim(REGIONS[0], "a")
    await board.note(REGIONS[0], "a", "finding", "x")
    assert await other.notes(REGIONS[0]) == []
    assert (await other.claim(REGIONS[0], "b"))["claimed"] is True


async def test_region_names_the_crosscheck_it_takes(pool, scope, caller):
    board, _ = await made(pool, scope)
    tools = board_tools(board)
    seen = await board.second()
    assert seen == SECOND
    inside = await caller(tools, "region", {"region": REGIONS[3]})
    outside = await caller(tools, "region", {"region": REGIONS[0]})
    assert inside["crosscheck"] == "crosscheck_second"
    assert outside["crosscheck"] == "crosscheck"
    assert outside["neighbours"] == list(moon.region(REGIONS[0]).neighbours)
    assert set(outside["bounds"]) == {"lat_min", "lat_max", "lon_min", "lon_max"}
    assert tuple(outside["centre"]) == moon.region(REGIONS[0]).centre


async def test_counts_follow_claims_and_landings(pool, scope):
    board, _ = await made(pool, scope, regions=REGIONS[:10])
    assert await board.counts() == {"landed": 0, "failed": 0, "claimed": 0, "free": 10}
    await board.claim(REGIONS[0], "a")
    await board.claim(REGIONS[1], "b")
    assert await board.counts() == {"landed": 0, "failed": 0, "claimed": 2, "free": 8}
    await board.land(row(REGIONS[0], "a"))
    assert await board.counts() == {"landed": 1, "failed": 0, "claimed": 1, "free": 8}
    assert [r["region"] for r in await board.landed()] == [REGIONS[0]]


async def test_the_tools_act_for_the_run_that_calls(pool, scope, caller):
    board, _ = await made(pool, scope)
    tools = board_tools(board)
    assert sorted(tools.names) == ["claim", "dossier", "note", "notes", "region"]
    got = await caller(tools, "claim", {"region": REGIONS[0]}, "a")
    assert got == {"claimed": True, "by": "a"}
    await caller(tools, "note", {"region": REGIONS[0], "kind": "finding", "text": "x"}, "a")
    listed = await caller(tools, "notes", {"region": REGIONS[0]}, "b")
    assert [n["author"] for n in listed["notes"]] == ["a"]


async def test_the_tools_refuse_outside_a_run(pool, scope):
    board, _ = await made(pool, scope)
    with pytest.raises(BoardError, match="board: no run is calling"):
        await board_tools(board).dispatch("claim", {"region": REGIONS[0]})


async def test_counts_partition_the_regions_in_every_state(pool, scope):
    flight = Flight("live", "again", "twice")
    board, _ = await made(pool, scope, flight, regions=REGIONS[:8])
    r = REGIONS
    await board.claim(r[0], "live")
    await board.claim(r[1], "dead")
    await board.claim(r[2], "x")
    await board.fail(r[2], "x", "ran out")
    await board.claim(r[3], "x")
    await board.fail(r[3], "x", "ran out")
    await board.claim(r[3], "again")
    await board.claim(r[4], "gone")
    await board.land(row(r[4], "gone"))
    await board.land(row(r[5], "twice"))
    await board.land(row(r[5], "other"))
    counts = await board.counts()
    assert counts == {"landed": 2, "claimed": 2, "failed": 1, "free": 3}
    assert sum(counts.values()) == 8


async def test_a_claim_waits_for_a_landing_in_progress(pool, scope):
    flight = Flight("a", "b")
    board, _ = await made(pool, scope, flight)
    await board.claim(REGIONS[0], "a")
    flight.running.discard("a")
    async with pool.acquire() as conn:
        tx = conn.transaction()
        await tx.start()
        await conn.fetchval(
            "select 1 from claims where scope = $1 and survey = 's' and region = $2 for update",
            scope,
            REGIONS[0],
        )
        racing = asyncio.create_task(board.claim(REGIONS[0], "b"))
        await asyncio.sleep(0.2)
        assert not racing.done()
        await conn.execute(
            "insert into landed (scope, survey, region, run_id, provider, simulated,"
            " dossier, checks, data_check, truth, disputes)"
            " values ($1, 's', $2, 'a', 'fake', true, '{}', '{}', '{}', '{}', '[]')",
            scope,
            REGIONS[0],
        )
        await tx.commit()
    assert await racing == {"claimed": False, "by": "a"}


async def test_a_claim_over_a_row_that_a_failure_removes_still_owns_it(pool, scope):
    flight = Flight("a", "b")
    board, _ = await made(pool, scope, flight)
    await board.claim(REGIONS[0], "a")
    async with pool.acquire() as conn:
        tx = conn.transaction()
        await tx.start()
        await conn.execute(
            "delete from claims where scope = $1 and survey = 's' and region = $2",
            scope,
            REGIONS[0],
        )
        racing = asyncio.create_task(board.claim(REGIONS[0], "b"))
        await asyncio.sleep(0.2)
        await tx.commit()
    assert await racing == {"claimed": True, "by": "b"}
    assert await board.claim_of("b") == REGIONS[0]


async def test_a_note_outside_the_survey_is_refused(pool, scope):
    board, _ = await made(pool, scope, regions=REGIONS[:10])
    with pytest.raises(BoardError, match=f"board: region '{REGIONS[20]}' is not in survey 's'"):
        await board.note(REGIONS[20], "a", "finding", "t")


async def test_a_landing_waits_for_the_claim_row_lock(pool, scope):
    board, _ = await made(pool, scope)
    await board.claim(REGIONS[0], "a")
    async with pool.acquire() as conn:
        tx = conn.transaction()
        await tx.start()
        await conn.fetchval(
            "select 1 from claims where scope = $1 and survey = 's' and region = $2 for update",
            scope,
            REGIONS[0],
        )
        landing = asyncio.create_task(board.land(row(REGIONS[0], "a")))
        await asyncio.sleep(0.2)
        assert not landing.done()
        await tx.commit()
    assert await landing is True


async def test_the_dossier_tool_reads_the_landed_dossier_beside_the_measured_values(
    pool, scope, caller, monkeypatch
):
    monkeypatch.setenv("MOON_SURVEY_DATA", str(CUT))
    board, _ = await made(pool, scope)
    tools = board_tools(board)
    region = "S75-90/0"
    before = await caller(tools, "dossier", {"region": region})
    assert before["region"] == region and before["landed"] is None
    measured = moon.measured(region)
    assert before["measured"]["FeO"] == {
        "mean": measured["FeO"].mean,
        "error": measured["FeO"].error,
        "pixels": measured["FeO"].pixels,
        "unit": "wt%",
    }
    assert before["measured"]["H"]["unit"] == "ppm" and before["measured"]["Th"]["unit"] == "ppm"
    await board.claim(region, "a")
    await board.land(row(region, dossier={"region": region, "values": {}}))
    after = await caller(tools, "dossier", {"region": region})
    assert after["landed"] == {"region": region, "values": {}}
    assert set(after) == {"region", "landed", "measured"}
    empty = await caller(tools, "dossier", {"region": "N60-75/5"})
    assert empty["measured"]["FeO"] is None


async def test_a_note_is_written_on_the_connection_that_creates_the_survey(pool, scope):
    async with pool.acquire() as conn, conn.transaction():
        await create_survey(
            conn,
            scope,
            "s",
            5.0,
            set(),
            REGIONS[:2],
            mode="hostile",
            shape="crews",
            crews=1,
            status="running",
        )
        note = await write_note(conn, scope, "s", REGIONS[0], "planted", "finding", "x")
        assert await pool.fetchval("select count(*) from notes where scope = $1", scope) == 0
    found = await Board(pool, scope, "s", live=Flight()).notes(REGIONS[0])
    assert [(n["id"], n["author"], n["kind"]) for n in found] == [(note, "planted", "finding")]
    with pytest.raises(BoardError, match="board: region .* is not in survey 's'"):
        await write_note(pool, scope, "s", REGIONS[5], "planted", "finding", "x")


async def test_failed_lists_each_region_with_a_failure_once(pool, scope):
    board, _ = await made(pool, scope)
    assert await board.failed() == set()
    await board.fail(REGIONS[0], "a", "first")
    await board.fail(REGIONS[0], "b", "second")
    await board.fail(REGIONS[1], "a", "third")
    assert await board.failed() == {REGIONS[0], REGIONS[1]}

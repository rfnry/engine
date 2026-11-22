import asyncio
import secrets

import pytest
from rfnry_engine import CallInfo

import board as board_module
import moon
from board import Board, BoardError, board_tools, create_survey

REGIONS = [r.id for r in moon.regions()]
SECOND = {REGIONS[3], REGIONS[4]}


@pytest.fixture
def scope():
    return f"t{secrets.token_hex(4)}"


class Flight:
    def __init__(self, *running):
        self.running = set(running)

    def __call__(self, run_id):
        return run_id in self.running


async def made(pool, scope, flight=None, regions=None, second=SECOND):
    flight = flight or Flight("a", "b", "c")
    await create_survey(pool, scope, "s", 5.0, second, regions or REGIONS)
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
    await create_survey(pool, scope, "other", 1.0, set(), REGIONS)
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


async def test_the_tools_act_for_the_run_that_calls(pool, scope, caller):
    board, _ = await made(pool, scope)
    tools = board_tools(board)
    assert sorted(tools.names) == ["claim", "note", "notes", "region"]
    got = await caller(tools, "claim", {"region": REGIONS[0]}, "a")
    assert got == {"claimed": True, "by": "a"}
    await caller(tools, "note", {"region": REGIONS[0], "kind": "finding", "text": "x"}, "a")
    listed = await caller(tools, "notes", {"region": REGIONS[0]}, "b")
    assert [n["author"] for n in listed["notes"]] == ["a"]


async def test_the_tools_refuse_outside_a_run(pool, scope):
    board, _ = await made(pool, scope)
    with pytest.raises(BoardError, match="board: no run is calling"):
        await board_tools(board).dispatch("claim", {"region": REGIONS[0]})

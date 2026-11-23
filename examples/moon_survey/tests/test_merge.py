import asyncio
import copy
import importlib.util
import json
import secrets
from pathlib import Path

import pytest
from rfnry_engine import InputError, RunEnded, Usage
from rfnry_engine import Message as EngineMessage

import moon
from board import Board, create_survey
from merge import PRICES, Merge, Price, Spend, SpendError, export_map
from moon import Value

EXAMPLE = Path(__file__).resolve().parent.parent
CHECK = EXAMPLE / "agents" / "surveyor" / "skills" / "dossier" / "check.py"
CUT = EXAMPLE / "tests" / "data" / "lp_cut"
GRS = (
    "https://pds-geosciences.wustl.edu/lunar/lp-l-grs-5-elem-abundance-v1/lp_9001/data/"
    "lpgrs_high1_elem_abundance_2deg.tab"
)


def load_problems():
    spec = importlib.util.spec_from_file_location("dossier_check", CHECK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.problems


PROBLEMS = load_problems()


@pytest.fixture
def scope():
    return f"t{secrets.token_hex(4)}"


class Places:
    def __init__(self):
        self.files = {}

    async def read_at(self, run_id, path, limit):
        found = self.files.get(run_id, {}).get(path)
        if found is None:
            raise FileNotFoundError(path)
        if len(found) > limit:
            raise InputError(f"read: {path} is over {limit} bytes")
        return found

    def put(self, run_id, dossier, path="dossier.json"):
        text = dossier if isinstance(dossier, (bytes, str)) else json.dumps(dossier)
        self.files.setdefault(run_id, {})[path] = text.encode() if isinstance(text, str) else text


class Fetches:
    def __init__(self, **urls):
        self.found = {k: set(v) for k, v in urls.items()}

    async def urls(self, run_id):
        return self.found.get(run_id, set())


class Flight:
    def __init__(self):
        self.gone = set()

    def __call__(self, run_id):
        return run_id not in self.gone


def value(mean, error, pixels):
    return Value(mean, error, pixels, mean, mean, mean, mean)


def stub_measured(region):
    return {
        "FeO": value(8.0, 1.0, 10),
        "TiO2": value(0.6, 0.2, 10),
        "Al2O3": value(20.0, 1.0, 10),
        "Th": value(1.0, 0.5, 10),
        "H": value(100.0, 0.0, 10),
    }


def site(region, name="Apollo X", **values):
    return {
        "site": name,
        "region": region,
        "values": {k: {"value": v, "unit": "wt%", "citation": {}} for k, v in values.items()},
    }


async def made(
    pool,
    scope,
    regions,
    truth=(),
    measured=stub_measured,
    fetches=None,
    places=None,
    provider="anthropic",
):
    await create_survey(
        pool, scope, "s", 5.0, set(), regions, mode="fake", shape="crews", crews=1, status="running"
    )
    board = Board(pool, scope, "s", live=Flight())
    places = places or Places()
    merge = Merge(
        board,
        places,
        fetches or Fetches(),
        truth=list(truth),
        measured=measured,
        problems=PROBLEMS,
        provider=lambda agent: provider,
    )
    return board, places, merge


def ended(run_id, agent="surveyor", error=None, stopped=None):
    return RunEnded(run_id=run_id, agent=agent, call=None, stopped=stopped, error=error)


async def failures(pool, scope):
    rows = await pool.fetch(
        "select region, run_id, reason from failures where scope = $1 order by at, region", scope
    )
    return [dict(r) for r in rows]


def of(dossier, region):
    found = copy.deepcopy(dossier)
    found["region"] = region
    return found


REGION = "N75-90/0"


async def test_a_clean_dossier_lands(pool, scope, dossier):
    fetches = Fetches(a={GRS})
    board, places, merge = await made(pool, scope, [REGION], fetches=fetches)
    await board.claim(REGION, "a")
    places.put("a", dossier)
    await merge.on_event(ended("a"))
    [row] = await board.landed(REGION)
    assert row["run_id"] == "a"
    assert row["provider"] == "anthropic"
    assert row["simulated"] is False
    assert row["checks"] == {"problems": []}
    assert row["disputes"] == []
    marks = {
        name: [s["mark"] for s in value["sources"]]
        for name, value in row["dossier"]["values"].items()
        if "sources" in value
    }
    assert marks == {
        "water_ice": ["dataset"],
        "TiO2": ["dataset"],
        "FeO": ["dataset"],
        "Al2O3": ["dataset"],
        "Th": ["dataset"],
    }
    assert row["dossier"]["values"]["He3"] == dossier["values"]["He3"]
    assert row["data_check"]["verdicts"] == {
        "water_ice": "pass",
        "TiO2": "pass",
        "FeO": "pass",
        "Al2O3": "fail",
        "Th": "pass",
        "He3": "unchecked",
    }
    assert row["data_check"]["measured"]["FeO"] == {"mean": 8.0, "error": 1.0, "pixels": 10}
    assert row["data_check"]["measured"]["WEH"] == pytest.approx(0.09)
    assert row["truth"] is None
    assert await failures(pool, scope) == []


async def test_sources_are_marked_by_what_they_are(pool, scope, dossier):
    fetches = Fetches(a={GRS, "https://arxiv.org/abs/1"})
    board, places, merge = await made(pool, scope, [REGION], fetches=fetches)
    await board.claim(REGION, "a")
    mixed = copy.deepcopy(dossier)
    mixed["values"]["FeO"]["sources"] = [
        GRS,
        "https://arxiv.org/abs/1",
        "data/lp/hydrogenlow.txt",
        "a textbook",
        "https://pds-geosciences.wustl.edu/missions/lunarp/reduced/hydrogenlow.txt",
    ]
    places.put("a", mixed)
    await merge.on_event(ended("a"))
    [row] = await board.landed(REGION)
    assert row["dossier"]["values"]["FeO"]["sources"] == [
        {"source": GRS, "mark": "dataset"},
        {"source": "https://arxiv.org/abs/1", "mark": "fetched"},
        {"source": "data/lp/hydrogenlow.txt", "mark": "dataset"},
        {"source": "a textbook", "mark": "cited"},
        {
            "source": "https://pds-geosciences.wustl.edu/missions/lunarp/reduced/hydrogenlow.txt",
            "mark": "dataset",
        },
    ]


async def test_a_missing_or_failing_dossier_fails_its_region(pool, scope, dossier):
    regions = [REGION, "S75-90/0", "S60-75/0", "S60-75/1", "N60-75/0"]
    board, places, merge = await made(pool, scope, regions)
    for run_id, region in zip("abcde", regions, strict=True):
        await board.claim(region, run_id)
    bad = copy.deepcopy(dossier)
    del bad["values"]["FeO"]
    places.put("b", "{not json")
    places.put("c", bad)
    places.put("d", of(dossier, "S75-90/0"))
    for run_id in "abcd":
        await merge.on_event(ended(run_id))
    await merge.on_event(ended("e", error="ModelError: down"))
    found = {f["region"]: f["reason"] for f in await failures(pool, scope)}
    assert found[REGION] == "merge: no dossier.json"
    assert found["S75-90/0"].startswith("merge: dossier.json is not valid JSON")
    assert found["S60-75/0"].startswith("merge: dossier.json fails its checks: ")
    assert "FeO" in found["S60-75/0"]
    assert found["S60-75/1"].startswith("merge: dossier.json is for region 'S75-90/0'")
    assert found["N60-75/0"] == "ModelError: down"
    assert await board.landed() == []


async def test_a_dossier_over_the_limit_fails(pool, scope, dossier):
    board, places, merge = await made(pool, scope, [REGION])
    await board.claim(REGION, "a")
    places.put("a", b" " * 2_000_000)
    await merge.on_event(ended("a"))
    [found] = await failures(pool, scope)
    assert found["reason"].startswith("merge: dossier.json cannot be read: ")
    assert "over" in found["reason"]


async def test_a_stopped_surveyor_still_lands_the_dossier_it_finished(pool, scope, dossier):
    board, places, merge = await made(pool, scope, [REGION])
    await board.claim(REGION, "a")
    places.put("a", dossier)
    await merge.on_event(ended("a", error="ModelError: late"))
    assert [r["region"] for r in await board.landed()] == [REGION]


async def test_a_surveyor_without_a_claim_and_other_agents_do_nothing(pool, scope, dossier):
    board, places, merge = await made(pool, scope, [REGION])
    places.put("a", dossier)
    await merge.on_event(ended("a"))
    await merge.on_event(ended("a", agent="boss"))
    await board.claim(REGION, "b")
    places.put("b", dossier)
    await merge.on_event(ended("b", agent="boss"))
    assert await board.landed() == []
    assert await failures(pool, scope) == []


async def test_a_truth_region_is_compared(pool, scope, dossier):
    truth = [site("N00-15/18", TiO2=7.54, FeO=15.81, Al2O3=13.55, Th=1.94)]
    board, places, merge = await made(pool, scope, ["N00-15/18"], truth=truth)
    await board.claim("N00-15/18", "a")
    found = of(dossier, "N00-15/18")
    found["values"]["TiO2"].update(low=7, high=8)
    found["values"]["FeO"].update(low=4, high=7)
    found["values"]["Al2O3"] = {"unknown": "no source"}
    found["values"]["Th"].update(low=1, high=3)
    places.put("a", found)
    await merge.on_event(ended("a"))
    [row] = await board.landed()
    assert row["truth"]["site"] == "Apollo X"
    assert {k: v["verdict"] for k, v in row["truth"]["values"].items()} == {
        "TiO2": "pass",
        "FeO": "fail",
        "Al2O3": "missing",
        "Th": "pass",
    }
    assert row["truth"]["values"]["FeO"]["truth"] == 15.81
    assert "water_ice" not in row["truth"]["values"]


@pytest.mark.parametrize(
    ("mean", "error", "verdict"),
    [(8.0, 1.0, "pass"), (8.0, 0.1, "fail"), (10.0, 1.5, "pass"), (3.0, 0.4, "fail")],
)
async def test_the_data_check_allows_the_datasets_error(pool, scope, dossier, mean, error, verdict):
    def measured(region):
        return {**stub_measured(region), "FeO": value(mean, error, 4)}

    board, places, merge = await made(pool, scope, [REGION], measured=measured)
    await board.claim(REGION, "a")
    places.put("a", dossier)
    await merge.on_event(ended("a"))
    [row] = await board.landed()
    assert row["data_check"]["verdicts"]["FeO"] == verdict


async def test_water_is_checked_one_sided_against_hydrogen(pool, scope, dossier):
    def measured(region):
        return {**stub_measured(region), "H": value(200.0, 0.0, 4)}

    board, places, merge = await made(pool, scope, [REGION, "S75-90/0"], measured=measured)
    await board.claim(REGION, "a")
    await board.claim("S75-90/0", "b")
    wet = copy.deepcopy(dossier)
    wet["values"]["water_ice"].update(low=0.2, high=3.0)
    places.put("a", wet)
    higher = of(dossier, "S75-90/0")
    higher["values"]["water_ice"].update(low=0.17, high=3.0)
    places.put("b", higher)
    await merge.on_event(ended("a"))
    await merge.on_event(ended("b"))
    first, second = await board.landed(REGION), await board.landed("S75-90/0")
    assert first[0]["data_check"]["measured"]["WEH"] == pytest.approx(0.18)
    assert first[0]["data_check"]["verdicts"]["water_ice"] == "fail"
    assert second[0]["data_check"]["verdicts"]["water_ice"] == "pass"


async def test_a_value_the_dataset_lacks_is_unchecked(pool, scope, dossier):
    def measured(region):
        return {k: None for k in stub_measured(region)}

    board, places, merge = await made(pool, scope, [REGION], measured=measured)
    await board.claim(REGION, "a")
    places.put("a", dossier)
    await merge.on_event(ended("a"))
    [row] = await board.landed()
    assert set(row["data_check"]["verdicts"].values()) == {"unchecked"}
    assert row["data_check"]["measured"]["FeO"] is None


async def test_a_dispute_marks_its_value_and_lowers_its_confidence(pool, scope, dossier):
    board, places, merge = await made(pool, scope, [REGION])
    await board.claim(REGION, "a")
    places.put("a", dossier)
    await merge.on_event(ended("a"))
    await board.note(REGION, "x", "agree", "TiO2: the range matches the GRS map")
    await board.note(REGION, "x", "dispute", "FeO: the cited table is for another region")
    await board.note(REGION, "x", "dispute", "the whole dossier cites one page")
    await board.note(REGION, "y", "dispute", "Th: not mine")
    await merge.on_event(ended("x", agent="crosscheck"))
    [row] = await board.landed()
    values = row["dossier"]["values"]
    assert values["FeO"]["disputed"] == "the cited table is for another region"
    assert values["FeO"]["confidence"] == "low"
    assert values["FeO"]["low"] == 4
    assert "disputed" not in values["TiO2"]
    assert values["TiO2"]["confidence"] == "medium"
    assert "disputed" not in values["Th"]
    assert [(d["kind"], d["name"]) for d in row["disputes"]] == [
        ("agree", "TiO2"),
        ("dispute", "FeO"),
        ("dispute", None),
    ]
    assert row["disputes"][2]["text"] == "the whole dossier cites one page"
    await merge.on_event(ended("x", agent="crosscheck"))
    [again] = await board.landed()
    assert again["disputes"] == row["disputes"]


async def test_a_crosscheck_that_ends_before_the_landing_is_applied_when_it_lands(
    pool, scope, dossier
):
    board, places, merge = await made(pool, scope, [REGION])
    await board.claim(REGION, "a")
    await board.note(REGION, "x", "dispute", "Th: wrong")
    await merge.on_event(ended("x", agent="crosscheck_second"))
    places.put("a", dossier)
    await merge.on_event(ended("a"))
    [row] = await board.landed()
    assert row["dossier"]["values"]["Th"]["disputed"] == "wrong"
    assert row["dossier"]["values"]["Th"]["confidence"] == "low"


async def test_a_note_from_a_run_that_is_no_crosscheck_changes_nothing(pool, scope, dossier):
    board, places, merge = await made(pool, scope, [REGION])
    await board.claim(REGION, "a")
    places.put("a", dossier)
    await board.note(REGION, "z", "dispute", "FeO: wrong")
    await merge.on_event(ended("a"))
    await merge.on_event(ended("z", agent="boss"))
    [row] = await board.landed()
    assert row["disputes"] == []
    assert "disputed" not in row["dossier"]["values"]["FeO"]


def reply(model, **usage):
    return EngineMessage(role="assistant", content="x", model=model, usage=Usage(**usage))


class Presses:
    def __init__(self):
        self.count = 0

    def after_step(self):
        self.count += 1


async def spend_of(pool, scope, cap=5.0, stop=None):
    spend = Spend(pool, scope, "s", cap, stop or Presses())
    await spend.load()
    return spend


async def test_a_reply_is_priced_by_its_model(pool, scope):
    spend = await spend_of(pool, scope)
    await spend.reply(
        scope,
        "c",
        reply("claude-haiku-5-5", input=1000, output=500, cache_read=2000, cache_creation=400),
    )
    assert spend.total == pytest.approx(
        (1000 * 0.10 + 2000 * 0.01 + 400 * 0.125 + 500 * 0.50) / 1e6
    )
    before = spend.total
    await spend.reply(scope, "c", reply("claude-haiku-5-5", input=100_000, output=10))
    assert spend.total - before == pytest.approx((100_000 * 0.10 + 10 * 0.50) / 1e6)
    before = spend.total
    await spend.reply(
        scope, "c", reply("claude-haiku-5-5", input=60_000, output=10, cache_read=40_001)
    )
    assert spend.total - before == pytest.approx((60_000 * 0.50 + 40_001 * 0.05 + 10 * 2.50) / 1e6)
    before = spend.total
    for model in ("gemini-3.5-flash-lite", "jev-latest", "fake"):
        await spend.reply(scope, "c", reply(model, input=10**6, output=10**6))
    assert spend.total == before


def test_the_prices_are_the_plans():
    assert PRICES["claude-haiku-5-5"] == Price(0.10, 0.50, 0.50, 2.50, 100_000)
    assert set(PRICES) == {"claude-haiku-5-5", "gemini-3.5-flash-lite", "jev-latest", "fake"}


async def test_an_unknown_model_is_refused(pool, scope):
    spend = await spend_of(pool, scope)
    with pytest.raises(SpendError, match="^spend: no price for model 'gpt-x'"):
        await spend.reply(scope, "c", reply("gpt-x", input=1))
    with pytest.raises(SpendError, match="^spend: no price for model 'None'"):
        await spend.reply(scope, "c", EngineMessage(role="assistant", usage=Usage(input=1)))
    assert spend.total == 0


async def test_a_decision_is_priced_like_a_reply(pool, scope):
    spend = await spend_of(pool, scope)
    await spend.decision("claude-haiku-5-5", Usage(input=1000, output=100))
    assert spend.total == pytest.approx((1000 * 0.10 + 100 * 0.50) / 1e6)


async def test_the_cap_presses_after_step_once(pool, scope):
    stop = Presses()
    spend = await spend_of(pool, scope, cap=0.001, stop=stop)
    each = reply("claude-haiku-5-5", input=4000, output=0)
    await spend.reply(scope, "c", each)
    await spend.reply(scope, "c", each)
    assert stop.count == 0
    await spend.reply(scope, "c", each)
    assert stop.count == 1
    await spend.reply(scope, "c", each)
    await spend.decision("claude-haiku-5-5", Usage(input=10))
    assert stop.count == 1
    assert spend.total == pytest.approx(4 * 4000 * 0.10 / 1e6 + 10 * 0.10 / 1e6)


async def test_replies_landing_together_are_all_kept_and_press_once(pool, scope):
    stop = Presses()
    spend = await spend_of(pool, scope, cap=0.01, stop=stop)
    each = reply("claude-haiku-5-5", input=1000, output=0)
    await asyncio.gather(*(spend.reply(scope, "c", each) for _ in range(250)))
    assert spend.total == pytest.approx(250 * 1000 * 0.10 / 1e6)
    assert stop.count == 1
    assert (await spend_of(pool, scope)).total == pytest.approx(spend.total)


async def test_the_spend_survives_a_restart(pool, scope):
    first = await spend_of(pool, scope)
    await first.reply(scope, "c", reply("claude-haiku-5-5", input=100_000))
    again = await spend_of(pool, scope)
    assert again.total == pytest.approx(0.01)


async def test_a_spend_loaded_at_its_cap_presses_on_its_first_reply(pool, scope):
    first = await spend_of(pool, scope, cap=0.01)
    await first.reply(scope, "c", reply("claude-haiku-5-5", input=100_000))
    stop = Presses()
    again = await spend_of(pool, scope, cap=0.01, stop=stop)
    assert stop.count == 0
    await again.reply(scope, "c", reply("fake", input=1))
    assert stop.count == 1
    await again.reply(scope, "c", reply("fake", input=1))
    assert stop.count == 1


async def test_the_map_holds_every_measured_region_and_no_simulated_dossier(
    pool, scope, dossier, tmp_path, monkeypatch
):
    monkeypatch.setenv("MOON_SURVEY_DATA", str(CUT))
    inside = sorted(moon.pixels_by_region(CUT))[:3]
    regions = [*inside, "N75-90/0"]
    board, places, merge = await made(pool, scope, regions)
    for run_id, region in zip("abc", [*inside[:2], "N75-90/0"], strict=True):
        await board.claim(region, run_id)
    places.put("a", of(dossier, inside[0]))
    simulated = of(dossier, inside[1])
    simulated["simulated"] = True
    places.put("b", simulated)
    places.put("c", dossier)
    for run_id in "abc":
        await merge.on_event(ended(run_id))
    path = tmp_path / "map" / "moon.json"
    await export_map(board, path)
    text = path.read_text()
    assert text.endswith("}\n")
    found = json.loads(text)
    assert list(found) == ["survey", "made", "regions"]
    assert found["survey"] == "s"
    assert [r["region"] for r in found["regions"]] == regions
    assert list(found["regions"][0]) == [
        "region",
        "bounds",
        "measured",
        "dossier",
        "checks",
        "data_check",
        "truth",
        "disputes",
        "provider",
    ]
    by = {r["region"]: r for r in found["regions"]}
    assert by[inside[0]]["dossier"]["region"] == inside[0]
    assert by[inside[0]]["provider"] == "anthropic"
    assert by[inside[1]]["dossier"] is None
    assert by[inside[1]]["data_check"] is None
    assert by[inside[1]]["measured"] == {
        k: (None if v is None else {"mean": v.mean, "error": v.error, "pixels": v.pixels})
        for k, v in moon.measured(inside[1]).items()
    }
    assert by[inside[2]]["dossier"] is None
    assert by[inside[2]]["provider"] is None
    assert by[inside[0]]["bounds"] == {
        "lat_min": moon.region(inside[0]).lat_min,
        "lat_max": moon.region(inside[0]).lat_max,
        "lon_min": moon.region(inside[0]).lon_min,
        "lon_max": moon.region(inside[0]).lon_max,
    }
    assert "true" not in json.dumps([by[inside[1]]["dossier"]])
    assert list(path.parent.iterdir()) == [path]


async def test_a_dossier_of_nested_brackets_fails_its_region_and_raises_nothing(pool, scope):
    board, places, merge = await made(pool, scope, [REGION])
    await board.claim(REGION, "a")
    places.put("a", b"[" * 200_000)
    await merge.on_event(ended("a"))
    [found] = await failures(pool, scope)
    assert found["reason"].startswith("merge: dossier.json cannot be checked: RecursionError")


async def test_a_stopped_run_with_no_error_and_no_dossier_fails_its_region(pool, scope):
    board, places, merge = await made(pool, scope, [REGION])
    await board.claim(REGION, "a")
    await merge.on_event(ended("a", stopped="now"))
    [found] = await failures(pool, scope)
    assert found["reason"] == "merge: no dossier.json"


async def test_a_checker_that_raises_fails_its_region(pool, scope, dossier):
    board, places, merge = await made(pool, scope, [REGION])
    merge.problems = lambda found: 1 / 0
    await board.claim(REGION, "a")
    places.put("a", dossier)
    await merge.on_event(ended("a"))
    [found] = await failures(pool, scope)
    assert "ZeroDivisionError" in found["reason"]


async def test_a_fake_dossier_is_simulated_whatever_it_wrote(
    pool, scope, dossier, tmp_path, monkeypatch
):
    monkeypatch.setenv("MOON_SURVEY_DATA", str(CUT))
    board, places, merge = await made(pool, scope, [REGION], provider="fake")
    await board.claim(REGION, "a")
    places.put("a", dossier)
    await merge.on_event(ended("a"))
    [row] = await board.landed()
    assert row["simulated"] is True
    await export_map(board, tmp_path / "moon.json")
    [shown] = json.loads((tmp_path / "moon.json").read_text())["regions"]
    assert shown["dossier"] is None


async def test_a_failed_attempts_crosscheck_dispute_does_not_reach_the_next_landing(
    pool, scope, dossier
):
    board, places, merge = await made(pool, scope, [REGION])
    flight = board.live
    await board.claim(REGION, "a")
    await board.note(REGION, "x", "dispute", "FeO: wrong")
    await merge.on_event(ended("x", agent="crosscheck"))
    await merge.on_event(ended("a"))
    flight.gone.add("a")
    await board.claim(REGION, "b")
    places.put("b", dossier)
    await merge.on_event(ended("b"))
    [row] = await board.landed()
    assert row["run_id"] == "b"
    assert row["disputes"] == []
    assert "disputed" not in row["dossier"]["values"]["FeO"]


async def test_the_prompt_counts_cache_creation_toward_the_threshold(pool, scope):
    spend = await spend_of(pool, scope)
    await spend.reply(
        scope,
        "c",
        reply("claude-haiku-5-5", input=50_000, cache_read=40_000, cache_creation=10_001),
    )
    assert spend.total == pytest.approx((50_000 * 0.50 + 40_000 * 0.05 + 10_001 * 0.625) / 1e6)


async def test_two_exports_at_once_do_not_collide(pool, scope, tmp_path, monkeypatch):
    monkeypatch.setenv("MOON_SURVEY_DATA", str(CUT))
    board, places, merge = await made(pool, scope, sorted(moon.pixels_by_region(CUT))[:2])
    path = tmp_path / "moon.json"
    await asyncio.gather(export_map(board, path), export_map(board, path))
    assert list(tmp_path.iterdir()) == [path]
    assert len(json.loads(path.read_text())["regions"]) == 2


@pytest.mark.parametrize("step", ["marked", "data_check", "truth_check"])
async def test_a_step_over_a_dossier_that_raises_fails_its_region(
    pool, scope, dossier, monkeypatch, step
):
    import merge as merge_module

    def broken(*args):
        raise KeyError("boom")

    monkeypatch.setattr(merge_module, step, broken)
    board, places, merge = await made(pool, scope, [REGION])
    await board.claim(REGION, "a")
    places.put("a", dossier)
    await merge.on_event(ended("a"))
    [found] = await failures(pool, scope)
    assert found["reason"].startswith("merge: dossier.json cannot be merged: KeyError")
    assert await board.landed() == []

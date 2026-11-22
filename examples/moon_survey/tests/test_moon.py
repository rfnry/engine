import json
from pathlib import Path

import pytest

import moon

BANDS = {"00-15": 33, "15-30": 31, "30-45": 27, "45-60": 20, "60-75": 13, "75-90": 1}
ROOT = Path(__file__).resolve().parent.parent


def test_the_moon_holds_250_regions():
    found = moon.regions()
    assert len(found) == 250
    assert len({r.id for r in found}) == 250


@pytest.mark.parametrize("hemisphere", ["N", "S"])
def test_each_hemisphere_has_its_bands(hemisphere):
    counts: dict[str, int] = {}
    for r in moon.regions():
        band, _ = r.id.split("/")
        if band[0] == hemisphere:
            counts[band[1:]] = counts.get(band[1:], 0) + 1
    assert counts == BANDS


def test_cells_are_near_equal_in_area():
    areas = [r.area for r in moon.regions() if not r.id.endswith("75-90/0")]
    assert len(areas) == 248
    assert max(areas) / min(areas) <= 1.05


def test_neighbours_are_mutual():
    by_id = {r.id: r for r in moon.regions()}
    for r in by_id.values():
        assert r.id not in r.neighbours
        for other in r.neighbours:
            assert r.id in by_id[other].neighbours


def test_a_cap_neighbours_every_cell_of_its_band():
    for cap, band in (("N75-90/0", "N60-75"), ("S75-90/0", "S60-75")):
        cells = {r.id for r in moon.regions() if r.id.startswith(band + "/")}
        assert len(cells) == 13
        assert set(moon.region(cap).neighbours) == cells


def test_region_of_finds_the_cell():
    assert moon.region_of(0.674, 23.473) == "N00-15/18"
    assert moon.region_of(-89.0, 10.0) == "S75-90/0"
    assert moon.region_of(5.0, 180.0) == moon.region_of(5.0, -180.0)


def test_shares_cut_the_moon_into_ten_connected_crews():
    crews = moon.shares()
    assert len(crews) == 10
    assert all(len(c) == 25 for c in crews)
    flat = [i for c in crews for i in c]
    assert len(set(flat)) == 250
    assert set(flat) == {r.id for r in moon.regions()}
    for crew in crews:
        members = set(crew)
        seen = {crew[0]}
        todo = [crew[0]]
        while todo:
            for n in moon.region(todo.pop()).neighbours:
                if n in members and n not in seen:
                    seen.add(n)
                    todo.append(n)
        assert seen == members


def test_regions_json_matches_the_code(tmp_path):
    target = tmp_path / "regions.json"
    moon.write_regions(target)
    assert (ROOT / "data" / "regions.json").read_text() == target.read_text()
    assert json.loads(target.read_text())[0]["id"]


def test_region_of_returns_every_region_at_its_centre_and_its_western_edge():
    for r in moon.regions():
        lat, lon = r.centre
        assert moon.region_of(lat, lon) == r.id
        assert moon.region_of(lat, r.lon_min) == r.id


@pytest.mark.parametrize(
    ("lat", "expected"),
    [
        (0, "N00-15/0"),
        (-0.0, "N00-15/0"),
        (15, "N15-30/0"),
        (-15, "S15-30/0"),
        (30, "N30-45/0"),
        (-30, "S30-45/0"),
        (45, "N45-60/0"),
        (-45, "S45-60/0"),
        (60, "N60-75/0"),
        (-60, "S60-75/0"),
        (75, "N75-90/0"),
        (-75, "S75-90/0"),
        (90, "N75-90/0"),
        (-90, "S75-90/0"),
    ],
)
def test_a_latitude_edge_falls_in_the_band_away_from_the_equator_and_the_poles_in_the_caps(
    lat, expected
):
    assert moon.region_of(lat, -180.0) == expected


def test_region_of_refuses_what_is_not_a_place():
    for lat, lon in ((float("nan"), 0.0), (0.0, float("inf")), (91.0, 0.0)):
        with pytest.raises(ValueError, match="^moon: "):
            moon.region_of(lat, lon)


def test_an_unknown_region_says_so():
    with pytest.raises(LookupError, match=r"^moon: region 'X9/1' does not exist$"):
        moon.region("X9/1")


def test_the_grid_cannot_be_corrupted_by_a_caller():
    moon.regions().clear()
    assert len(moon.regions()) == 250
    with pytest.raises(TypeError):
        moon.by_id()["N00-15/0"] = None

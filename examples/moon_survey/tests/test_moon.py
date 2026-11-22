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

import json
from pathlib import Path

import moon

TRUTH = Path(__file__).resolve().parent.parent / "data" / "truth.json"
UNIT_OF = {
    "TiO2": "wt%",
    "FeO": "wt%",
    "Al2O3": "wt%",
    "water_ice": "wt%",
    "Th": "ppm",
    "He3": "ppb",
}
SOILS = [
    "Apollo 11",
    "Apollo 12",
    "Apollo 15",
    "Apollo 16",
    "Apollo 17",
    "Luna 16",
    "Luna 24",
    "Chang'e-5",
]


def sites():
    return json.loads(TRUTH.read_text())


def test_nine_sites_each_in_its_own_region():
    found = sites()
    assert len(found) == 9
    for site in found:
        assert site["region"] == moon.region_of(site["lat"], site["lon"])
    assert len({site["region"] for site in found}) == 9


def test_every_value_has_a_unit_and_a_citation():
    for site in sites():
        assert site["values"]
        for name, entry in site["values"].items():
            assert entry["unit"] == UNIT_OF[name], (site["site"], name)
            assert isinstance(entry["value"], (int, float))
            citation = entry["citation"]
            assert citation["source"] and citation["where"]
            assert citation["url"].startswith("https://")


def test_the_soil_sites_carry_the_major_oxides():
    by_site = {site["site"]: site for site in sites()}
    for name in SOILS:
        assert {"TiO2", "FeO", "Al2O3"} <= set(by_site[name]["values"]), name


def test_cabeus_carries_water():
    by_site = {site["site"]: site for site in sites()}
    water = by_site["Cabeus"]["values"]["water_ice"]
    assert water["unit"] == "wt%"


def test_every_position_is_cited_and_in_range():
    for site in sites():
        assert -90 <= site["lat"] <= 90
        assert -180 <= site["lon"] <= 180
        position = site["position"]
        assert position["source"] and position["where"]
        assert position["url"].startswith("https://")


def test_cabeus_water_is_a_percentage():
    by_site = {site["site"]: site for site in sites()}
    value = by_site["Cabeus"]["values"]["water_ice"]["value"]
    assert isinstance(value, (int, float)) and 0 < value < 100


def test_every_soil_says_what_maturity_its_source_gives():
    by_site = {site["site"]: site for site in sites()}
    for name in SOILS:
        for entry in by_site[name]["values"].values():
            assert "Maturity:" in entry["citation"]["where"], name

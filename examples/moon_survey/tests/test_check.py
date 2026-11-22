import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import jsonschema_rs
import pytest

SKILL = Path(__file__).resolve().parent.parent / "agents" / "surveyor" / "skills" / "dossier"
RANGES = {
    "water_ice": (0, 100, "wt%"),
    "TiO2": (0, 20, "wt%"),
    "FeO": (0, 30, "wt%"),
    "Al2O3": (0, 40, "wt%"),
    "Th": (0, 20, "ppm"),
    "He3": (0, 100, "ppb"),
}


def load_check():
    spec = importlib.util.spec_from_file_location("dossier_check", SKILL / "check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def check():
    return load_check()


def change(path, value):
    def apply(dossier):
        *above, last = path
        target = dossier
        for key in above:
            target = target[key]
        if value is DROP:
            del target[last]
        else:
            target[last] = value

    return apply


DROP = object()
CASES = {
    "low above high": (
        [change(("values", "TiO2", "low"), 1.0), change(("values", "TiO2", "high"), 0.2)],
        ["values.TiO2: low 1.0 is above high 0.2"],
    ),
    "no source": (
        [change(("values", "FeO", "sources"), [])],
        ["values.FeO: sources is empty; name at least one source"],
    ),
    "out of range": (
        [change(("values", "Th", "high"), 25)],
        ["values.Th: high 25 is outside 0 to 20 ppm"],
    ),
    "below range": (
        [change(("values", "water_ice", "low"), -1)],
        ["values.water_ice: low -1 is outside 0 to 100 wt%"],
    ),
    "unknown confidence": (
        [change(("values", "Al2O3", "confidence"), "certain")],
        ['values.Al2O3: confidence "certain" is not one of low, medium, high'],
    ),
    "no region's form": (
        [change(("region",), "north pole")],
        ['region: "north pole" is not a region id of the form N00-15/3'],
    ),
    "no region of the grid": (
        [change(("region",), "N00-15/40")],
        ['region: "N00-15/40" is no region of the grid'],
    ),
    "unknown terrain": (
        [change(("terrain",), "crater")],
        ['terrain: "crater" is not one of mare, highland, polar, mixed'],
    ),
    "a missing value": (
        [change(("values", "He3"), DROP)],
        ['values: He3 is missing; give its range, or {"unknown": "<why>"}'],
    ),
    "a value of no dossier": (
        [change(("values", "CaO"), {"unknown": "not asked"})],
        [
            'values: "CaO" is not a value of a dossier; its values are '
            "water_ice, TiO2, FeO, Al2O3, Th, He3"
        ],
    ),
    "a source over http": (
        [change(("values", "TiO2", "sources"), ["http://arxiv.org/abs/1"])],
        [
            'values.TiO2: source "http://arxiv.org/abs/1" is a URL; a URL source starts with https://'
        ],
    ),
    "an unknown with no reason": (
        [change(("values", "He3", "unknown"), " ")],
        ["values.He3: unknown must say why, in a non-empty string"],
    ),
    "a range and an unknown at once": (
        [change(("values", "Th", "unknown"), "unsure")],
        ['values.Th: is a range or {"unknown": "<why>"}, not both'],
    ),
    "a missing source list": (
        [change(("values", "FeO", "sources"), DROP)],
        ["values.FeO: sources is missing"],
    ),
    "a field of no value": (
        [change(("values", "FeO", "mean"), 5)],
        [
            'values.FeO: "mean" is not a field of a value; '
            "its fields are low, high, confidence, sources"
        ],
    ),
    "a low that is no number": (
        [change(("values", "TiO2", "low"), "0.2")],
        ["values.TiO2: low must be a number in wt%"],
    ),
    "a high that is not finite": (
        [change(("values", "TiO2", "high"), float("nan"))],
        ["values.TiO2: high must be a number in wt%"],
    ),
    "a value that is no object": (
        [change(("values", "FeO"), 5)],
        ['values.FeO: must be {"low", "high", "confidence", "sources"} or {"unknown": "<why>"}'],
    ),
    "a missing field": (
        [change(("notes",), DROP)],
        ["dossier: notes is missing"],
    ),
    "a field of no dossier": (
        [change(("score",), 1)],
        [
            'dossier: "score" is not a field of a dossier; its fields are '
            "region, terrain, values, notes, simulated"
        ],
    ),
    "notes that are no text": (
        [change(("notes",), ["ice"])],
        ["notes: must be a string"],
    ),
    "simulated that is no boolean": (
        [change(("simulated",), "no")],
        ["simulated: must be true or false"],
    ),
}


def test_check_passes_a_clean_dossier(check, dossier):
    assert check.problems(dossier) == []


@pytest.mark.parametrize("case", list(CASES))
def test_check_names_each_problem(check, dossier, case):
    changes, expected = CASES[case]
    for apply in changes:
        apply(dossier)
    assert check.problems(dossier) == expected


def test_check_names_a_dossier_that_is_no_object(check):
    assert check.problems(["N00-15/3"]) == [
        "dossier: must be a JSON object of region, terrain, values, notes, simulated"
    ]


def test_check_names_every_problem_of_a_dossier_at_once(check, dossier):
    dossier["terrain"] = "crater"
    dossier["values"]["TiO2"]["sources"] = []
    assert check.problems(dossier) == [
        'terrain: "crater" is not one of mare, highland, polar, mixed',
        "values.TiO2: sources is empty; name at least one source",
    ]


def run_check(*args):
    return subprocess.run(
        [sys.executable, str(SKILL / "check.py"), *args], capture_output=True, text=True
    )


def test_check_runs_as_a_script(tmp_path, dossier):
    path = tmp_path / "dossier.json"
    path.write_text(json.dumps(dossier))
    clean = run_check(str(path))
    assert (clean.returncode, clean.stdout, clean.stderr) == (0, "", "")
    dossier["terrain"] = "crater"
    dossier["values"]["Th"]["high"] = 25
    path.write_text(json.dumps(dossier))
    done = run_check(str(path))
    assert (done.returncode, done.stderr) == (1, "")
    assert done.stdout.splitlines() == [
        'terrain: "crater" is not one of mare, highland, polar, mixed',
        "values.Th: high 25 is outside 0 to 20 ppm",
    ]


def test_check_names_a_file_it_cannot_read_or_parse(tmp_path):
    broken = tmp_path / "dossier.json"
    broken.write_text('{"region": ')
    done = run_check(str(broken))
    assert (done.returncode, done.stderr) == (1, "")
    assert done.stdout.startswith(f"{broken}: is not JSON: Expecting value")
    assert len(done.stdout.splitlines()) == 1
    missing = run_check(str(tmp_path / "none.json"))
    assert (missing.returncode, missing.stderr) == (1, "")
    assert (
        missing.stdout == f"{tmp_path / 'none.json'}: cannot be read: No such file or directory\n"
    )


def test_the_schema_holds_the_names_units_and_ranges(check, dossier):
    schema = json.loads((SKILL / "dossier.schema.json").read_text())
    assert jsonschema_rs.meta.is_valid(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    values = schema["properties"]["values"]
    assert values["required"] == list(RANGES)
    for name, (low, high, unit) in RANGES.items():
        spec = values["properties"][name]
        assert spec["unit"] == unit
        for end in ("low", "high"):
            assert spec["properties"][end] == {"minimum": low, "maximum": high}
    assert "top metre" in values["properties"]["water_ice"]["description"]
    validator = jsonschema_rs.Draft202012Validator(schema)
    assert validator.is_valid(dossier)
    for case in ("out of range", "unknown confidence", "no region's form", "no source"):
        broken = json.loads(json.dumps(dossier))
        for apply in CASES[case][0]:
            apply(broken)
        assert not validator.is_valid(broken), case

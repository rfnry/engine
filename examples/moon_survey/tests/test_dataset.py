import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

import moon
import setup

CUT = Path(__file__).parent / "data" / "lp_cut"
GRS = "lpgrs_high1_elem_abundance_2deg.tab"
HYDROGEN = "hydrogenlow.txt"
CUT_REGIONS = ("N75-90/0", "S75-90/0", "N60-75/0")


@pytest.fixture
def cut(monkeypatch):
    monkeypatch.setenv("MOON_SURVEY_DATA", str(CUT))
    return CUT


def field(line: str, start: int) -> float:
    return float(line[start - 1 : start + 14])


def grs_rows() -> list[dict]:
    rows = []
    for line in (CUT / GRS).read_text().splitlines():
        rows.append(
            {
                "lat": (float(line[10:17]) + float(line[17:24])) / 2,
                "lon": (float(line[24:31]) + float(line[31:38])) / 2,
                "Al2O3": (field(line, 84) * 100, math.sqrt(field(line, 339)) * 100),
                "TiO2": (field(line, 129) * 100, math.sqrt(field(line, 654)) * 100),
                "FeO": (field(line, 144) * 100, math.sqrt(field(line, 729)) * 100),
                "Th": (field(line, 174), math.sqrt(field(line, 834))),
            }
        )
    return rows


def hydrogen_rows() -> list[dict]:
    rows = []
    for line in (CUT / HYDROGEN).read_text().splitlines():
        if line.startswith("#"):
            continue
        low, high, west, east, value = (float(x) for x in line.split(","))
        rows.append({"lat": (low + high) / 2, "lon": (west + east) / 2, "H": (value, 0.0)})
    return rows


def test_a_pixel_reads_as_its_row(cut):
    raw = grs_rows()
    read = moon.read_grs(cut)
    assert len(read) == len(raw) > 300
    for expected, got in ((raw[0], read[0]), (raw[137], read[137]), (raw[-1], read[-1])):
        assert (got.lat, got.lon) == (expected["lat"], expected["lon"])
        for key in ("FeO", "TiO2", "Al2O3", "Th"):
            assert got.values[key] == pytest.approx(expected[key], rel=1e-9)
    hydrogen = moon.read_hydrogen(cut)
    assert [(p.lat, p.lon, p.values["H"]) for p in hydrogen] == [
        (r["lat"], r["lon"], r["H"]) for r in hydrogen_rows()
    ]


@pytest.mark.parametrize("region_id", CUT_REGIONS)
def test_a_region_is_the_mean_of_its_pixels(cut, region_id):
    found = moon.measured(region_id)
    assert set(found) == {"FeO", "TiO2", "Al2O3", "Th", "H"}
    for key, rows in (
        *((k, grs_rows()) for k in ("FeO", "TiO2", "Al2O3", "Th")),
        ("H", hydrogen_rows()),
    ):
        inside = [r[key] for r in rows if moon.region_of(r["lat"], r["lon"]) == region_id]
        got = found[key]
        assert got.pixels == len(inside) > 20
        assert got.mean == pytest.approx(sum(v for v, _ in inside) / len(inside))
        assert got.error == pytest.approx(math.sqrt(sum(e * e for _, e in inside)) / len(inside))
    assert found["H"].error == 0.0


def test_a_region_with_no_pixel_reads_none(cut):
    assert moon.measured("N00-15/0") == {
        "FeO": None,
        "TiO2": None,
        "Al2O3": None,
        "Th": None,
        "H": None,
    }


def test_an_unknown_region_is_refused(cut):
    with pytest.raises(moon.RegionError, match="moon: region 'nowhere' does not exist"):
        moon.measured("nowhere")


def test_the_data_directory_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("MOON_SURVEY_DATA", raising=False)
    assert moon.data_dir() == Path(moon.__file__).resolve().parent.parent / "data" / "lp"
    monkeypatch.setenv("MOON_SURVEY_DATA", str(tmp_path))
    assert moon.data_dir() == tmp_path


@pytest.mark.parametrize("missing", [GRS, HYDROGEN])
def test_a_missing_file_says_how_to_get_it(tmp_path, monkeypatch, missing):
    monkeypatch.setenv("MOON_SURVEY_DATA", str(tmp_path))
    for name in (GRS, HYDROGEN):
        if name != missing:
            (tmp_path / name).write_bytes((CUT / name).read_bytes())
    with pytest.raises(
        moon.DataError, match=f"moon: data/lp/{missing} is missing; run uv run src/setup.py data"
    ):
        moon.measured("N75-90/0")


def sources_table(entries: dict[str, bytes], digests: dict[str, str] | None = None) -> str:
    lines = [
        "# Sources",
        "",
        "| file | url | size | sha256 | note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, body in entries.items():
        digest = (digests or {}).get(name, hashlib.sha256(body).hexdigest())
        lines.append(f"| `{name}` | https://example.test/{name} | {len(body)} | `{digest}` | x |")
    return "\n".join(lines) + "\n"


def test_download_fetches_and_verifies(tmp_path):
    entries = {"a.bin": b"alpha", "b.bin": b"beta"}
    sources = tmp_path / "SOURCES.md"
    sources.write_text(sources_table(entries))
    asked = []

    def opener(url):
        asked.append(url)
        return entries[url.rsplit("/", 1)[1]]

    into = tmp_path / "lp"
    got = setup.download(sources, into, opener)
    assert got == [into / "a.bin", into / "b.bin"]
    assert (into / "b.bin").read_bytes() == b"beta"
    assert asked == ["https://example.test/a.bin", "https://example.test/b.bin"]


def test_download_refuses_a_wrong_checksum(tmp_path):
    body = b"alpha"
    claimed = "0" * 64
    sources = tmp_path / "SOURCES.md"
    sources.write_text(sources_table({"a.bin": body}, {"a.bin": claimed}))
    into = tmp_path / "lp"
    with pytest.raises(setup.SetupError) as raised:
        setup.download(sources, into, lambda url: body)
    assert str(raised.value) == (
        f"setup: a.bin has sha256 {hashlib.sha256(body).hexdigest()}; SOURCES.md says {claimed}"
    )
    assert not (into / "a.bin").exists()


def test_download_replaces_an_existing_corrupt_file(tmp_path):
    sources = tmp_path / "SOURCES.md"
    sources.write_text(sources_table({"a.bin": b"alpha"}))
    into = tmp_path / "lp"
    into.mkdir()
    (into / "a.bin").write_bytes(b"alp")
    setup.download(sources, into, lambda url: b"alpha")
    assert (into / "a.bin").read_bytes() == b"alpha"
    assert sorted(p.name for p in into.iterdir()) == ["a.bin"]


def test_download_refuses_a_wrong_size(tmp_path):
    body = b"alpha"
    sources = tmp_path / "SOURCES.md"
    sources.write_text(sources_table({"a.bin": body}).replace(f"| {len(body)} |", "| 99 |"))
    into = tmp_path / "lp"
    with pytest.raises(setup.SetupError) as raised:
        setup.download(sources, into, lambda url: body)
    assert str(raised.value) == "setup: a.bin has 5 bytes; SOURCES.md says 99"
    assert list(into.iterdir()) == []


def test_a_failed_write_leaves_nothing_behind(tmp_path, monkeypatch):
    sources = tmp_path / "SOURCES.md"
    sources.write_text(sources_table({"a.bin": b"alpha"}))
    into = tmp_path / "lp"

    def failing(self, data):
        self.touch()
        raise OSError("no space left on device")

    monkeypatch.setattr(Path, "write_bytes", failing)
    with pytest.raises(OSError, match="no space left"):
        setup.download(sources, into, lambda url: b"alpha")
    assert list(into.iterdir()) == []


def test_download_skips_a_file_already_there(tmp_path):
    sources = tmp_path / "SOURCES.md"
    sources.write_text(sources_table({"a.bin": b"alpha"}))
    into = tmp_path / "lp"
    into.mkdir()
    (into / "a.bin").write_bytes(b"alpha")

    def opener(url):
        raise AssertionError(f"fetched {url}")

    assert setup.download(sources, into, opener) == [into / "a.bin"]


def test_the_real_sources_list_the_files_the_loader_reads():
    listed = setup.parse_sources(Path(moon.__file__).resolve().parent.parent / "data/lp/SOURCES.md")
    assert {s.name for s in listed} == {GRS, HYDROGEN}
    for source in listed:
        assert source.url.startswith("https://pds-geosciences.wustl.edu/")
        assert len(source.sha256) == 64
        assert source.size > 0


def test_a_malformed_row_says_where(tmp_path, monkeypatch):
    monkeypatch.setenv("MOON_SURVEY_DATA", str(tmp_path))
    rows = (CUT / GRS).read_text().splitlines()
    (tmp_path / GRS).write_text("\n".join([*rows[:2], rows[2][:60], *rows[3:]]))
    (tmp_path / HYDROGEN).write_bytes((CUT / HYDROGEN).read_bytes())
    with pytest.raises(moon.DataError, match=f"moon: {GRS} line 3 does not parse: "):
        moon.measured("N75-90/0")


def test_a_malformed_hydrogen_row_says_where(tmp_path, monkeypatch):
    monkeypatch.setenv("MOON_SURVEY_DATA", str(tmp_path))
    (tmp_path / GRS).write_bytes((CUT / GRS).read_bytes())
    lines = (CUT / HYDROGEN).read_text().splitlines()
    lines[13] = "  -90.00, nope"
    (tmp_path / HYDROGEN).write_text("\n".join(lines))
    with pytest.raises(moon.DataError, match=f"moon: {HYDROGEN} line 14 does not parse: "):
        moon.measured("N75-90/0")


def spread_of(inside):
    values = sorted(v for v, _ in inside)
    deciles = statistics.quantiles(values, n=10, method="inclusive")
    return {"min": values[0], "p10": deciles[0], "p90": deciles[8], "max": values[-1]}


def region_rows(key, region_id):
    rows = hydrogen_rows() if key == "H" else grs_rows()
    return [r[key] for r in rows if moon.region_of(r["lat"], r["lon"]) == region_id]


@pytest.mark.parametrize("region_id", CUT_REGIONS)
def test_a_region_carries_the_spread_of_its_pixels(cut, region_id):
    found = moon.measured(region_id)
    for key in ("FeO", "TiO2", "Al2O3", "Th", "H"):
        expected = spread_of(region_rows(key, region_id))
        got = found[key]
        assert (got.min, got.p10, got.p90, got.max) == pytest.approx(
            (expected["min"], expected["p10"], expected["p90"], expected["max"])
        )
        assert got.min <= got.p10 <= got.p90 <= got.max
        assert got.min < got.max


@pytest.mark.parametrize("region_id", CUT_REGIONS)
def test_the_measure_script_prints_the_spread_beside_the_mean(cut, monkeypatch, region_id):
    src = Path(moon.__file__).resolve().parent
    script = src.parent / "agents/surveyor/skills/dossier/measure.py"
    monkeypatch.setenv("MOON_SURVEY_SRC", str(src))
    done = subprocess.run(
        [sys.executable, str(script), region_id], capture_output=True, text=True, check=True
    )
    printed = json.loads(done.stdout)["values"]
    for key in ("FeO", "TiO2", "Al2O3", "Th", "H"):
        inside = region_rows(key, region_id)
        expected = spread_of(inside)
        assert set(printed[key]) == {"mean", "error", "pixels", "min", "p10", "p90", "max"}
        for name, value in expected.items():
            assert printed[key][name] == pytest.approx(value)

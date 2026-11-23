import json
import math
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType

COUNTS = (33, 31, 27, 20, 13, 1)
STEP = 15
GRS_FILE = "lpgrs_high1_elem_abundance_2deg.tab"
HYDROGEN_FILE = "hydrogenlow.txt"
UNITS = {"FeO": "wt%", "TiO2": "wt%", "Al2O3": "wt%", "Th": "ppm", "H": "ppm"}
GRS_COLUMNS = {
    "Al2O3": (8, 25, 100.0),
    "TiO2": (11, 46, 100.0),
    "FeO": (12, 51, 100.0),
    "Th": (14, 58, 1.0),
}


@dataclass(frozen=True)
class Region:
    id: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    neighbours: tuple[str, ...]

    @property
    def centre(self) -> tuple[float, float]:
        return (self.lat_min + self.lat_max) / 2, (self.lon_min + self.lon_max) / 2

    @property
    def area(self) -> float:
        height = math.sin(math.radians(self.lat_max)) - math.sin(math.radians(self.lat_min))
        return math.radians(self.lon_max - self.lon_min) * height


@dataclass(frozen=True)
class Value:
    mean: float
    error: float
    pixels: int
    min: float
    p10: float
    p90: float
    max: float


@dataclass(frozen=True)
class Pixel:
    lat: float
    lon: float
    values: dict[str, tuple[float, float]]


class RegionError(LookupError):
    pass


class DataError(FileNotFoundError):
    pass


def band_name(hemisphere: str, index: int) -> str:
    return f"{hemisphere}{index * STEP:02d}-{(index + 1) * STEP:02d}"


def cell_id(key: tuple[str, int], cell: int) -> str:
    return f"{band_name(*key)}/{cell}"


def band_order() -> list[tuple[str, int]]:
    south = [("S", i) for i in range(len(COUNTS) - 1, -1, -1)]
    north = [("N", i) for i in range(len(COUNTS))]
    return south + north


def cell_span(index: int, cell: int) -> tuple[float, float]:
    width = 360 / COUNTS[index]
    return -180 + cell * width, -180 + (cell + 1) * width


def overlaps(first: tuple[float, float], second: tuple[float, float]) -> bool:
    return first[0] < second[1] - 1e-9 and second[0] < first[1] - 1e-9


@cache
def table() -> tuple[Region, ...]:
    bands = band_order()
    ids = {key: [cell_id(key, k) for k in range(COUNTS[key[1]])] for key in bands}
    links: dict[str, set[str]] = {i: set() for cells in ids.values() for i in cells}

    def join(a: str, b: str) -> None:
        if a != b:
            links[a].add(b)
            links[b].add(a)

    for key in bands:
        cells = ids[key]
        for k, cell in enumerate(cells):
            join(cell, cells[(k + 1) % len(cells)])
    for below, above in zip(bands, bands[1:], strict=False):
        for k, a in enumerate(ids[below]):
            for j, b in enumerate(ids[above]):
                if overlaps(cell_span(below[1], k), cell_span(above[1], j)):
                    join(a, b)

    found = []
    for key in bands:
        hemisphere, index = key
        low, high = index * STEP, (index + 1) * STEP
        lats = (float(low), float(high)) if hemisphere == "N" else (float(-high), float(-low))
        for k, cell in enumerate(ids[key]):
            lon_min, lon_max = cell_span(index, k)
            found.append(
                Region(cell, lats[0], lats[1], lon_min, lon_max, tuple(sorted(links[cell])))
            )
    return tuple(found)


def regions() -> list[Region]:
    return list(table())


@cache
def by_id() -> MappingProxyType[str, Region]:
    return MappingProxyType({r.id: r for r in table()})


def region(id: str) -> Region:
    try:
        return by_id()[id]
    except KeyError:
        raise RegionError(f"moon: region '{id}' does not exist") from None


def region_of(lat: float, lon: float) -> str:
    if not math.isfinite(lat) or not math.isfinite(lon):
        raise ValueError(f"moon: latitude {lat} and longitude {lon} must be finite")
    if not -90 <= lat <= 90:
        raise ValueError(f"moon: latitude {lat} is outside -90 to 90")
    hemisphere = "N" if lat >= 0 else "S"
    index = min(int(abs(lat) // STEP), len(COUNTS) - 1)
    if not -180 <= lon < 180:
        lon = (lon + 180) % 360 - 180
    cell = min(max(int((lon + 180) // (360 / COUNTS[index])), 0), COUNTS[index] - 1)
    while cell > 0 and lon < cell_span(index, cell)[0]:
        cell -= 1
    while cell < COUNTS[index] - 1 and lon >= cell_span(index, cell)[1]:
        cell += 1
    return cell_id((hemisphere, index), cell)


def shares(size: int = 25) -> list[list[str]]:
    walk: list[str] = []
    for n, key in enumerate(band_order()):
        cells = [cell_id(key, k) for k in range(COUNTS[key[1]])]
        walk.extend(cells if n % 2 == 0 else reversed(cells))
    return [walk[i : i + size] for i in range(0, len(walk), size)]


def data_dir() -> Path:
    given = os.environ.get("MOON_SURVEY_DATA")
    if given:
        return Path(given)
    return Path(__file__).resolve().parent.parent / "data" / "lp"


def read_lines(directory: Path, name: str) -> list[tuple[int, str]]:
    path = directory / name
    if not path.is_file():
        raise DataError(f"moon: data/lp/{name} is missing; run uv run src/setup.py data")
    return [(n, line) for n, line in enumerate(path.read_text().splitlines(), 1) if line.strip()]


def read_grs(directory: Path) -> list[Pixel]:
    found = []
    for n, line in read_lines(directory, GRS_FILE):
        cells = line.split()
        try:
            values = {
                key: (float(cells[value]) * scale, math.sqrt(float(cells[error])) * scale)
                for key, (value, error, scale) in GRS_COLUMNS.items()
            }
            lat = (float(cells[1]) + float(cells[2])) / 2
            lon = (float(cells[3]) + float(cells[4])) / 2
        except (IndexError, ValueError) as error:
            raise DataError(f"moon: {GRS_FILE} line {n} does not parse: {error}") from None
        found.append(Pixel(lat, lon, values))
    return found


def read_hydrogen(directory: Path) -> list[Pixel]:
    found = []
    for n, line in read_lines(directory, HYDROGEN_FILE):
        if line.startswith("#"):
            continue
        try:
            low, high, west, east, value = (float(x) for x in line.split(","))
        except ValueError as error:
            raise DataError(f"moon: {HYDROGEN_FILE} line {n} does not parse: {error}") from None
        found.append(Pixel((low + high) / 2, (west + east) / 2, {"H": (value, 0.0)}))
    return found


@cache
def pixels_by_region(directory: Path) -> MappingProxyType[str, tuple[Pixel, ...]]:
    grouped: dict[str, list[Pixel]] = {}
    for pixel in read_grs(directory) + read_hydrogen(directory):
        grouped.setdefault(region_of(pixel.lat, pixel.lon), []).append(pixel)
    return MappingProxyType({k: tuple(v) for k, v in grouped.items()})


def percentile(ordered: list[float], share: float) -> float:
    at = (len(ordered) - 1) * share
    below = math.floor(at)
    above = min(below + 1, len(ordered) - 1)
    return ordered[below] + (ordered[above] - ordered[below]) * (at - below)


def measured(region_id: str) -> dict[str, Value | None]:
    region(region_id)
    inside = pixels_by_region(data_dir()).get(region_id, ())
    found: dict[str, Value | None] = {}
    for key in ("FeO", "TiO2", "Al2O3", "Th", "H"):
        taken = [p.values[key] for p in inside if key in p.values]
        if not taken:
            found[key] = None
            continue
        n = len(taken)
        mean = sum(v for v, _ in taken) / n
        error = math.sqrt(sum(e * e for _, e in taken)) / n
        ordered = sorted(v for v, _ in taken)
        found[key] = Value(
            mean,
            error,
            n,
            ordered[0],
            percentile(ordered, 0.1),
            percentile(ordered, 0.9),
            ordered[-1],
        )
    return found


def write_regions(path: Path) -> None:
    rows = [
        {
            "id": r.id,
            "lat_min": r.lat_min,
            "lat_max": r.lat_max,
            "lon_min": r.lon_min,
            "lon_max": r.lon_max,
            "centre": list(r.centre),
            "neighbours": list(r.neighbours),
        }
        for r in table()
    ]
    path.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    write_regions(Path(__file__).resolve().parent.parent / "data" / "regions.json")

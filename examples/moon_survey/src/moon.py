import json
import math
from dataclasses import dataclass
from functools import cache
from pathlib import Path

COUNTS = (33, 31, 27, 20, 13, 1)
STEP = 15


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


def band_name(hemisphere: str, index: int) -> str:
    return f"{hemisphere}{index * STEP:02d}-{(index + 1) * STEP:02d}"


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
def regions() -> list[Region]:
    bands = band_order()
    ids = {key: [f"{band_name(*key)}/{k}" for k in range(COUNTS[key[1]])] for key in bands}
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
        lats = (low, high) if hemisphere == "N" else (-high, -low)
        for k, cell in enumerate(ids[key]):
            lon_min, lon_max = cell_span(index, k)
            found.append(
                Region(cell, lats[0], lats[1], lon_min, lon_max, tuple(sorted(links[cell])))
            )
    return found


@cache
def by_id() -> dict[str, Region]:
    return {r.id: r for r in regions()}


def region(id: str) -> Region:
    try:
        return by_id()[id]
    except KeyError:
        raise KeyError(f"moon: no region {id}") from None


def region_of(lat: float, lon: float) -> str:
    if not -90 <= lat <= 90:
        raise ValueError(f"moon: latitude {lat} is outside -90 to 90")
    hemisphere = "N" if lat >= 0 else "S"
    index = min(int(abs(lat) // STEP), len(COUNTS) - 1)
    wrapped = (lon + 180) % 360
    cell = min(int(wrapped // (360 / COUNTS[index])), COUNTS[index] - 1)
    return f"{band_name(hemisphere, index)}/{cell}"


def shares(size: int = 25) -> list[list[str]]:
    walk: list[str] = []
    for n, key in enumerate(band_order()):
        cells = [f"{band_name(*key)}/{k}" for k in range(COUNTS[key[1]])]
        walk.extend(cells if n % 2 == 0 else reversed(cells))
    return [walk[i : i + size] for i in range(0, len(walk), size)]


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
        for r in regions()
    ]
    path.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    write_regions(Path(__file__).resolve().parent.parent / "data" / "regions.json")

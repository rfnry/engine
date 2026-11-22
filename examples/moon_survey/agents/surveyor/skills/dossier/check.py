import importlib
import json
import math
import os
import re
import sys
from pathlib import Path

if source := os.environ.get("MOON_SURVEY_SRC"):
    sys.path.append(source)
moon = importlib.import_module("moon")

HERE = Path(__file__).resolve().parent
SCHEMA = json.loads((HERE / "dossier.schema.json").read_text(encoding="utf-8"))
FIELDS = list(SCHEMA["properties"])
REQUIRED = SCHEMA["required"]
REGION = re.compile(SCHEMA["properties"]["region"]["pattern"])
TERRAINS = SCHEMA["properties"]["terrain"]["enum"]
VALUES = SCHEMA["properties"]["values"]["properties"]
NAMES = SCHEMA["properties"]["values"]["required"]
RANGE = SCHEMA["$defs"]["range"]
RANGE_FIELDS = list(RANGE["properties"])
RANGE_REQUIRED = RANGE["required"]
CONFIDENCES = RANGE["properties"]["confidence"]["enum"]
SOURCES = RANGE["properties"]["sources"]
LEAST = SOURCES["minItems"]
HTTPS = re.compile(SOURCES["items"]["anyOf"][0]["pattern"])
URL = re.compile(SOURCES["items"]["anyOf"][1]["not"]["pattern"])
TEXT = re.compile(SCHEMA["$defs"]["text"]["pattern"])
GRID = frozenset(moon.by_id())
SHAPE = '{"low", "high", "confidence", "sources"} or {"unknown": "<why>"}'


def shown(value: object) -> str:
    if isinstance(value, dict):
        return "an object"
    if isinstance(value, list):
        return "a list"
    text = json.dumps(value)
    return text if len(text) <= 80 else text[:77] + "..."


def listed(names: list[str]) -> str:
    return ", ".join(names)


def region_problems(region: object) -> list[str]:
    if not isinstance(region, str) or not REGION.search(region):
        return [f"region: {shown(region)} is not a region id of the form N00-15/3"]
    if region not in GRID:
        return [f"region: {shown(region)} is no region of the grid"]
    return []


def number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value)


def range_problems(where: str, quantity: dict, spec: dict) -> list[str]:
    found = [f"{where}: {key} is missing" for key in RANGE_REQUIRED if key not in quantity]
    found += [
        f"{where}: {shown(key)} is not a field of a value; its fields are {listed(RANGE_FIELDS)}"
        for key in quantity
        if key not in RANGE_FIELDS
    ]
    for end in ("low", "high"):
        if end not in quantity:
            continue
        value, bounds = quantity[end], spec["properties"][end]
        if not number(value):
            found.append(f"{where}: {end} must be a number in {spec['unit']}")
        elif not bounds["minimum"] <= value <= bounds["maximum"]:
            found.append(
                f"{where}: {end} {shown(value)} is outside "
                f"{bounds['minimum']} to {bounds['maximum']} {spec['unit']}"
            )
    low, high = quantity.get("low"), quantity.get("high")
    if number(low) and number(high) and low > high:
        found.append(f"{where}: low {shown(low)} is above high {shown(high)}")
    if "confidence" in quantity and quantity["confidence"] not in CONFIDENCES:
        confidence = shown(quantity["confidence"])
        found.append(f"{where}: confidence {confidence} is not one of {listed(CONFIDENCES)}")
    if "sources" in quantity:
        found += source_problems(where, quantity["sources"])
    return found


def source_problems(where: str, sources: object) -> list[str]:
    if not isinstance(sources, list) or not all(
        isinstance(s, str) and TEXT.search(s) for s in sources
    ):
        return [f"{where}: sources must be a list of non-empty strings"]
    if len(sources) < LEAST:
        return [f"{where}: sources is empty; name at least one source"]
    return [
        f"{where}: source {shown(source)} is a URL; a URL source starts with https://"
        for source in sources
        if URL.search(source) and not HTTPS.search(source)
    ]


def quantity_problems(name: str, quantity: object) -> list[str]:
    where = f"values.{name}"
    if not isinstance(quantity, dict):
        return [f"{where}: must be {SHAPE}"]
    if "unknown" not in quantity:
        return range_problems(where, quantity, VALUES[name])
    if len(quantity) > 1:
        return [f'{where}: is a range or {{"unknown": "<why>"}}, not both']
    reason = quantity["unknown"]
    if not isinstance(reason, str) or not TEXT.search(reason):
        return [f"{where}: unknown must say why, in a non-empty string"]
    return []


def values_problems(values: object) -> list[str]:
    if not isinstance(values, dict):
        return [f"values: must be an object of {listed(NAMES)}"]
    found = [
        f'values: {name} is missing; give its range, or {{"unknown": "<why>"}}'
        for name in NAMES
        if name not in values
    ]
    found += [
        f"values: {shown(name)} is not a value of a dossier; its values are {listed(list(VALUES))}"
        for name in values
        if name not in VALUES
    ]
    for name in VALUES:
        if name in values:
            found += quantity_problems(name, values[name])
    return found


def problems(dossier: dict) -> list[str]:
    if not isinstance(dossier, dict):
        return [f"dossier: must be a JSON object of {listed(FIELDS)}"]
    found = [f"dossier: {name} is missing" for name in REQUIRED if name not in dossier]
    found += [
        f"dossier: {shown(name)} is not a field of a dossier; its fields are {listed(FIELDS)}"
        for name in dossier
        if name not in FIELDS
    ]
    if "region" in dossier:
        found += region_problems(dossier["region"])
    if "terrain" in dossier and dossier["terrain"] not in TERRAINS:
        found.append(f"terrain: {shown(dossier['terrain'])} is not one of {listed(TERRAINS)}")
    if "values" in dossier:
        found += values_problems(dossier["values"])
    if "notes" in dossier and not isinstance(dossier["notes"], str):
        found.append("notes: must be a string")
    if "simulated" in dossier and not isinstance(dossier["simulated"], bool):
        found.append("simulated: must be true or false")
    return found


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("check: usage: check.py dossier.json", file=sys.stderr)
        return 2
    try:
        dossier = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    except OSError as error:
        found = [f"{argv[0]}: cannot be read: {error.strerror}"]
    except (ValueError, RecursionError) as error:
        found = [f"{argv[0]}: is not JSON: {error}"]
    else:
        found = problems(dossier)
    for line in found:
        print(line)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

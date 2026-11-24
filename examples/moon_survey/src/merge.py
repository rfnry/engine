import asyncio
import json
import os
import re
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

import asyncpg
from rfnry_engine import Event, ReplyReady, RunEnded, Stop, Usage

import moon
from board import Board, BoardError
from sandbox import Sandboxes

SCHEMA = "moon"
SCOPE = "moon"
MAP_MODE = 0o644
USAGE_LINE = "usage: uv run src/merge.py <survey> <map.json>"
DOSSIER = "dossier.json"
DOSSIER_LIMIT = 1_000_000
VALUES = ("water_ice", "TiO2", "FeO", "Al2O3", "Th", "He3")
VERDICTS = ("agree", "dispute")
PER_MILLION = 1_000_000
CACHE_READ = 0.1
CACHE_CREATION = 1.25
LUNAR_PROSPECTOR = re.compile(r"lunar\s+prospector", re.IGNORECASE)
SPEND = """
insert into spend (scope, survey, total) values ($1, $2, $3)
on conflict (scope, survey) do update set total = spend.total + excluded.total
returning total
"""


class SpendError(Exception):
    pass


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    above_input: float
    above_output: float
    threshold: int


FREE = Price(0.0, 0.0, 0.0, 0.0, 100_000)
PRICES = {
    "claude-haiku-5-5": Price(0.10, 0.50, 0.50, 2.50, 100_000),
    "gemini-3.5-flash-lite": FREE,
    "jev-latest": FREE,
    "fake": FREE,
}


def cost_of(model: str | None, usage: Usage) -> float:
    price = PRICES.get(model) if model is not None else None
    if price is None:
        raise SpendError(f"spend: no price for model '{model}'")
    prompt = usage.input + usage.cache_read + usage.cache_creation
    high = prompt > price.threshold
    rate_in = price.above_input if high else price.input
    rate_out = price.above_output if high else price.output
    return (
        usage.input * rate_in
        + usage.cache_read * rate_in * CACHE_READ
        + usage.cache_creation * rate_in * CACHE_CREATION
        + usage.output * rate_out
    ) / PER_MILLION


class Spend:
    def __init__(self, pool: asyncpg.Pool, scope: str, survey: str, cap: float, stop: Stop) -> None:
        self.pool = pool
        self.scope = scope
        self.survey = survey
        self.cap = cap
        self.stop = stop
        self.total = 0.0
        self.pressed = False

    async def load(self) -> None:
        found = await self.pool.fetchval(
            "select total from spend where scope = $1 and survey = $2", self.scope, self.survey
        )
        self.total = found or 0.0

    async def replied(self, event: ReplyReady) -> None:
        added = cost_of(event.model, event.usage) if event.usage is not None else 0.0
        if added:
            self.total = max(
                self.total, await self.pool.fetchval(SPEND, self.scope, self.survey, added)
            )
        if self.total >= self.cap and not self.pressed:
            self.pressed = True
            self.stop.after_step()


@cache
def dataset_urls() -> frozenset[str]:
    return frozenset(source.url for source in moon.parse_sources(moon.SOURCES))


def mark(source: str, fetched: set[str]) -> dict[str, str]:
    if source in dataset_urls():
        return {"source": source, "mark": "dataset"}
    if source in fetched:
        return {"source": source, "mark": "fetched"}
    if source.startswith("https://"):
        return {"source": source, "mark": "cited"}
    if LUNAR_PROSPECTOR.search(source) or "data/lp" in source:
        return {"source": source, "mark": "dataset"}
    return {"source": source, "mark": "cited"}


def marked(dossier: dict[str, Any], fetched: set[str]) -> dict[str, Any]:
    found = json.loads(json.dumps(dossier))
    for value in found["values"].values():
        if "sources" in value:
            value["sources"] = [mark(source, fetched) for source in value["sources"]]
    return found


def data_check(dossier: dict[str, Any], found: Mapping[str, moon.Value | None]) -> dict[str, Any]:
    verdicts = {}
    for name in VALUES:
        value = dossier["values"][name]
        reference = found.get(name)
        if "unknown" in value or name == "He3" or (name != "water_ice" and reference is None):
            verdicts[name] = "unchecked"
        elif name == "water_ice":
            hydrogen = found.get("H")
            if hydrogen is None:
                verdicts[name] = "unchecked"
            else:
                weh = hydrogen.mean * moon.WEH_PER_H
                verdicts[name] = "pass" if value["low"] <= weh else "fail"
        else:
            assert reference is not None
            margin = moon.SIGMAS * reference.error
            near = (
                value["low"] <= reference.mean + margin and value["high"] >= reference.mean - margin
            )
            verdicts[name] = "pass" if near else "fail"
    hydrogen = found.get("H")
    used: dict[str, Any] = {
        name: moon.value_of(name, found.get(name)) for name in (*moon.MEASURED, "H")
    }
    used["WEH"] = None if hydrogen is None else hydrogen.mean * moon.WEH_PER_H
    return {"verdicts": verdicts, "measured": used}


def truth_check(
    dossier: dict[str, Any], sites: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    here = next((s for s in sites if s["region"] == dossier["region"]), None)
    if here is None:
        return None
    values = {}
    for name, known in here["values"].items():
        given = dossier["values"].get(name)
        if given is None or "unknown" in given:
            verdict = "missing"
        else:
            verdict = "pass" if given["low"] <= known["value"] <= given["high"] else "fail"
        values[name] = {"verdict": verdict, "truth": known["value"], "unit": known["unit"]}
    return {"site": here["site"], "values": values}


def disputed(dossier: dict[str, Any], name: str, reason: str) -> None:
    value = dossier["values"][name]
    earlier = value.get("disputed")
    value["disputed"] = reason if earlier is None else f"{earlier}; {reason}"
    if "confidence" in value:
        value["confidence"] = "low"


class Merge:
    def __init__(
        self,
        board: Board,
        sandboxes: Sandboxes,
        fetches: Any,
        *,
        truth: Sequence[Mapping[str, Any]],
        measured: Callable[[str], Mapping[str, moon.Value | None]],
        problems: Callable[[Any], list[str]],
        provider: Callable[[str], str],
    ) -> None:
        self.board = board
        self.sandboxes = sandboxes
        self.fetches = fetches
        self.truth = truth
        self.measured = measured
        self.problems = problems
        self.provider = provider
        self.crosschecks: set[str] = set()
        self.lock = asyncio.Lock()

    async def on_event(self, event: Event) -> None:
        if not isinstance(event, RunEnded) or event.paused:
            return
        if event.agent == "surveyor":
            await self.surveyor(event)
        elif event.agent in ("crosscheck", "crosscheck_second"):
            await self.crosscheck(event)

    async def read(self, run_id: str, region: str) -> tuple[dict[str, Any] | None, str | None]:
        try:
            body = await self.sandboxes.read_at(run_id, DOSSIER, DOSSIER_LIMIT)
        except FileNotFoundError:
            return None, f"merge: no {DOSSIER}"
        except Exception as error:
            return None, f"merge: {DOSSIER} cannot be read: {error}"
        try:
            dossier = json.loads(body)
        except ValueError as error:
            return None, f"merge: {DOSSIER} is not valid JSON: {error}"
        except Exception as error:
            return None, f"merge: {DOSSIER} cannot be checked: {type(error).__name__}: {error}"
        try:
            problems = self.problems(dossier)
            if problems:
                return None, f"merge: {DOSSIER} fails its checks: {'; '.join(problems)}"
            if dossier["region"] != region:
                return None, (
                    f"merge: {DOSSIER} is for region '{dossier['region']}'; the claim is '{region}'"
                )
        except Exception as error:
            return None, f"merge: {DOSSIER} cannot be checked: {type(error).__name__}: {error}"
        return dossier, None

    async def surveyor(self, event: RunEnded) -> None:
        region = await self.board.claim_of(event.run_id)
        if region is None:
            return
        dossier, reason = await self.read(event.run_id, region)
        if dossier is None:
            absent = reason == f"merge: no {DOSSIER}"
            await self.board.fail(
                region, event.run_id, event.error or reason or "" if absent else reason or ""
            )
            return
        fetched = await self.fetches.urls(event.run_id)
        async with self.lock:
            since = await self.board.claimed_at(event.run_id)
            notes = [
                n
                for n in await self.board.notes(region)
                if n["author"] in self.crosschecks and since is not None and n["at"] >= since
            ]
            try:
                row = self.row(event, region, dossier, fetched)
                for note in notes:
                    fold(row["dossier"], row["disputes"], note["author"], note)
            except Exception as error:
                reason = f"merge: {DOSSIER} cannot be merged: {type(error).__name__}: {error}"
                await self.board.fail(region, event.run_id, reason)
                return
            await self.board.land(row)

    def row(
        self, event: RunEnded, region: str, dossier: dict[str, Any], fetched: set[str]
    ) -> dict[str, Any]:
        provider = self.provider(event.agent)
        return {
            "region": region,
            "run_id": event.run_id,
            "provider": provider,
            "simulated": bool(dossier["simulated"]) or provider == "fake",
            "dossier": marked(dossier, fetched),
            "checks": {"problems": []},
            "data_check": data_check(dossier, self.measured(region)),
            "truth": truth_check(dossier, self.truth),
            "disputes": [],
        }

    async def crosscheck(self, event: RunEnded) -> None:
        async with self.lock:
            self.crosschecks.add(event.run_id)
            notes = await self.board.notes_by(event.run_id)
            for region in dict.fromkeys(note["region"] for note in notes):
                await self.attach(region, event.run_id, [n for n in notes if n["region"] == region])

    async def attach(self, region: str, run_id: str, notes: list[dict[str, Any]]) -> None:
        [row] = await self.board.landed(region) or [None]
        if row is None:
            return
        since = await self.board.claimed_at(row["run_id"])
        dossier, disputes = row["dossier"], list(row["disputes"])
        seen = {d["note"] for d in disputes}
        for note in notes:
            if note["id"] not in seen and since is not None and note["at"] >= since:
                fold(dossier, disputes, run_id, note)
        await self.board.update_landed(region, dossier=dossier, disputes=disputes)


def fold(
    dossier: dict[str, Any], disputes: list[dict[str, Any]], run_id: str, note: dict[str, Any]
) -> None:
    if note["kind"] not in VERDICTS:
        return
    name, _, rest = note["text"].partition(":")
    named = name in VALUES
    reason = rest.strip() if named else note["text"]
    disputes.append(
        {
            "note": note["id"],
            "run_id": run_id,
            "kind": note["kind"],
            "name": name if named else None,
            "text": reason,
        }
    )
    if named and note["kind"] == "dispute":
        disputed(dossier, name, reason)


def crosschecked(row: Mapping[str, Any]) -> bool:
    return any(found["kind"] in VERDICTS for found in row["disputes"])


async def map_of(board: Board) -> dict[str, Any]:
    landed = {row["region"]: row for row in await board.landed()}
    regions = []
    for region_id in await board.regions():
        row = landed.get(region_id)
        shown = row if row is not None and not row["simulated"] else None
        regions.append(
            {
                "region": region_id,
                "bounds": moon.region(region_id).bounds,
                "measured": {
                    name: moon.value_of(name, value)
                    for name, value in moon.measured(region_id).items()
                },
                "dossier": shown["dossier"] if shown else None,
                "checks": shown["checks"] if shown else None,
                "data_check": shown["data_check"] if shown else None,
                "truth": shown["truth"] if shown else None,
                "disputes": shown["disputes"] if shown else None,
                "provider": row["provider"] if row else None,
            }
        )
    return {
        "survey": board.survey,
        "made": datetime.now(UTC).isoformat(),
        "regions": regions,
    }


async def export_map(board: Board, path: Path) -> None:
    body = await map_of(board)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".part")
    partial = Path(name)
    try:
        with os.fdopen(handle, "w") as out:
            out.write(json.dumps(body, indent=2) + "\n")
            out.flush()
            os.fsync(out.fileno())
        partial.chmod(MAP_MODE)
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


async def exported(url: str, survey: str, path: Path, schema: str = SCHEMA) -> None:
    pool = await asyncpg.create_pool(url, server_settings={"search_path": schema})
    try:
        await export_map(Board(pool, SCOPE, survey, live=lambda _: False), path)
    finally:
        await pool.close()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(USAGE_LINE, file=sys.stderr)
        return 2
    url = os.environ.get("RFNRY_EXAMPLES_POSTGRES_URL")
    if not url:
        print("merge: RFNRY_EXAMPLES_POSTGRES_URL is not set", file=sys.stderr)
        return 1
    try:
        asyncio.run(exported(url, argv[0], Path(argv[1])))
    except (BoardError, OSError, asyncpg.PostgresError) as error:
        print(error, file=sys.stderr)
        return 1
    print(f"{argv[1]} ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

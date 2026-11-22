import asyncio
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import asyncpg
from rfnry_engine import Message, PostgresStorage, Text

import harness
import moon
from board import Board
from merge import cost_of

SCOPE = "moon"
LABELS = ("real", "fake", "arithmetic")
LEADS = ("lead", "flat", "captain", "boss")
USAGE = {"fake": "characters / 4, the fakes' estimate", "real": "as each provider reported it"}
REGION_ID = re.compile(r"[NS]\d\d-\d\d/\d+")
DIGITS = re.compile(r"(\d+)")
OURS = "(conversation = $2 or starts_with(conversation, $2 || '/'))"
SURVEY = "select mode, shape, status, regions from surveys where scope = $1 and survey = $2"
KEYS = f"select distinct conversation from messages where scope = $1 and {OURS}"
ROWS = f"""
select conversation, prompt, at, body::text as body from messages
where scope = $1 and {OURS} and (prompt is not null or conversation = any($3::text[]))
order by conversation, position
"""
SPAN = f"select min(at) as first, max(at) as last from messages where scope = $1 and {OURS}"
MOVES = f"select conversation, moves from windows where scope = $1 and {OURS}"
WAITS = """
select count(*) as count, coalesce(sum(wait), 0) as waited from provider_log
where scope = $1 and survey = $2 and status = 429
"""
GATE = """
select count(*) as calls, count(*) filter (where allowed is not true) as refused from gate_log
where scope = $1 and survey = $2
"""
SPENT = "select total from spend where scope = $1 and survey = $2"
TWICE = "select count(*) from landed where scope = $1 and survey = $2 and attempts > 1"
NOTES = "select region, author, text from notes where scope = $1 and survey = $2"
USAGE_LINE = "usage: uv run src/measure.py <survey> <results.md> [<title>]"


class MeasureError(Exception):
    pass


@dataclass(frozen=True)
class Stored:
    prompt: int | None
    at: datetime
    message: Message


def text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(part.text for part in content if isinstance(part, Text))


def data_of(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return None


def natural(key: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in DIGITS.split(key)]


def children(key: str, rows: list[Stored]) -> dict[str, str]:
    asked = {
        call.id: call.args.get("agent")
        for row in rows
        if row.message.role == "assistant"
        for call in row.message.tool_calls
        if call.name == "Delegate"
    }
    found = {}
    for row in rows:
        message = row.message
        if message.role != "tool" or message.is_error or message.tool_call_id not in asked:
            continue
        given = data_of(text_of(message.content))
        job = given.get("job") if isinstance(given, dict) else None
        agent = asked[message.tool_call_id]
        if isinstance(job, str) and isinstance(agent, str):
            found[f"{key}/{job}"] = agent
    return found


def wall(rows: list[Stored]) -> float:
    total, first, last = 0.0, None, None
    for row in rows:
        message = row.message
        if first is not None and message.role == "user" and message.notice is None:
            total += (last - first).total_seconds()
            first = None
        if first is None:
            first = row.at
        last = row.at
    if first is not None and last is not None:
        total += (last - first).total_seconds()
    return round(total, 3)


def size_of(result: Any) -> int:
    return len((result if isinstance(result, str) else json.dumps(result)).encode())


def said_by(dossier: dict[str, Any]) -> list[str]:
    said = [str(dossier.get("notes") or "")]
    for value in dossier.get("values", {}).values():
        for source in value.get("sources", []) if isinstance(value, dict) else []:
            said.append(str(source.get("source", "")) if isinstance(source, dict) else str(source))
    return said


def names_a_neighbour(region: str, dossier: dict[str, Any], notes: dict[str, list[str]]) -> bool:
    neighbours = set(moon.region(region).neighbours)
    said = said_by(dossier)
    if any(found in neighbours for text in said for found in REGION_ID.findall(text)):
        return True
    quoted = [note.strip() for n in neighbours for note in notes.get(n, []) if note.strip()]
    return any(note in text for note in quoted for text in said)


async def conversations(pool: asyncpg.Pool, name: str) -> tuple[set[str], dict[str, list[Stored]]]:
    keys = {row["conversation"] for row in await pool.fetch(KEYS, SCOPE, name)}
    parents = {key.rsplit("/", 1)[0] for key in keys}
    leads = {key for key in keys if key == name or key.startswith(harness.boss_prefix(name))}
    leads |= keys & parents
    stored: dict[str, list[Stored]] = {}
    for row in await pool.fetch(ROWS, SCOPE, name, sorted(leads)):
        stored.setdefault(row["conversation"], []).append(
            Stored(row["prompt"], row["at"], Message.model_validate_json(row["body"]))
        )
    return leads, stored


def agents_of(
    name: str, shape: str, leads: set[str], stored: dict[str, list[Stored]], moves: dict[str, int]
) -> dict[str, dict[str, Any]]:
    agents = {key: "boss" for key in leads if key.startswith(harness.boss_prefix(name))}
    if shape in harness.TOP:
        agents[name] = harness.TOP[shape]
    for key in leads:
        agents.update(children(key, stored.get(key, [])))
    table = {}
    for key in sorted(leads, key=natural):
        if agents.get(key) not in LEADS:
            continue
        rows = stored.get(key, [])
        prompt = [row.prompt for row in rows if row.prompt is not None]
        table[key] = {
            "agent": agents[key],
            "calls": len(prompt),
            "prompt": prompt,
            "wall_s": wall(rows),
            "moves": moves.get(key, 0),
        }
    return table


def by_model(stored: dict[str, list[Stored]]) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for rows in stored.values():
        for row in rows:
            usage = row.message.usage
            if row.message.role != "assistant" or usage is None:
                continue
            model = row.message.model
            entry = models.setdefault(
                str(model),
                {
                    "calls": 0,
                    "input": 0,
                    "cached": 0,
                    "cache_creation": 0,
                    "output": 0,
                    "spend": 0.0,
                },
            )
            entry["calls"] += 1
            entry["input"] += usage.input
            entry["cached"] += usage.cache_read
            entry["cache_creation"] += usage.cache_creation
            entry["output"] += usage.output
            entry["spend"] += cost_of(model, usage)
    return models


def lost_of(stored: dict[str, list[Stored]]) -> dict[str, Any]:
    lost = Counter(
        row.message.notice.agent
        for rows in stored.values()
        for row in rows
        if row.message.notice is not None and row.message.notice.kind == "lost"
    )
    return {"notices": sum(lost.values()), "agents": dict(lost)}


def jobs_of(records: list[Any]) -> dict[str, Any]:
    found = sorted(
        (step for record in records for step in record.body.get("steps", [])),
        key=lambda step: step["at"],
    )
    sizes = [size_of(step["result"]) for step in found if step["tool"] == "Jobs"]
    return {"calls": len(sizes), "bytes": sizes, "max": max(sizes, default=0), "total": sum(sizes)}


async def sharing(pool: asyncpg.Pool, name: str, records: list[Any]) -> dict[str, Any]:
    surveyors = [record for record in records if record.agent == "surveyor"]
    ids = {record.run_id for record in surveyors}
    read = [
        step
        for record in surveyors
        for step in record.body.get("steps", [])
        if step["tool"] == "notes" and not step["is_error"]
    ]
    notes = await pool.fetch(NOTES, SCOPE, name)
    return {
        "surveyors": len(surveyors),
        "written": sum(1 for note in notes if note["author"] in ids),
        "read": sum(len((data_of(step["result"]) or {}).get("notes", [])) for step in read),
        "read_calls": len(read),
    }


async def neighbours_named(pool: asyncpg.Pool, name: str, board: Board) -> dict[str, int]:
    on: dict[str, list[str]] = {}
    for note in await pool.fetch(NOTES, SCOPE, name):
        on.setdefault(note["region"], []).append(note["text"])
    landed = await board.landed()
    named = sum(1 for row in landed if names_a_neighbour(row["region"], row["dossier"], on))
    return {"landed": len(landed), "naming_a_neighbour": named}


async def measure(pool: asyncpg.Pool, engine_pool: asyncpg.Pool, name: str) -> dict[str, Any]:
    survey = await pool.fetchrow(SURVEY, SCOPE, name)
    if survey is None:
        raise MeasureError(f"measure: survey '{name}' does not exist")
    label = "fake" if survey["mode"] == "fake" else "real"
    leads, stored = await conversations(pool, name)
    moves = {row["conversation"]: row["moves"] for row in await pool.fetch(MOVES, SCOPE, name)}
    records = await PostgresStorage(engine_pool).run_tree(SCOPE, tags={"survey": name})
    board = Board(pool, SCOPE, name, live=lambda _: False)
    counts = await board.counts()
    span = await pool.fetchrow(SPAN, SCOPE, name)
    waits = await pool.fetchrow(WAITS, SCOPE, name)
    gated = await pool.fetchrow(GATE, SCOPE, name)
    return {
        "survey": name,
        "label": label,
        "mode": survey["mode"],
        "shape": survey["shape"],
        "status": survey["status"],
        "usage": USAGE[label],
        "regions": len(json.loads(survey["regions"])),
        "landed": counts["landed"],
        "failed": counts["failed"],
        "landed_twice": await pool.fetchval(TWICE, SCOPE, name),
        "wall_s": 0.0
        if span["first"] is None
        else round((span["last"] - span["first"]).total_seconds(), 3),
        "agents": agents_of(name, survey["shape"], leads, stored, moves),
        "jobs": jobs_of(records),
        "moves": sum(moves.values()),
        "rate_limits": {"429s": waits["count"], "waited_s": round(waits["waited"], 3)},
        "notes": await sharing(pool, name, records),
        "dossiers": await neighbours_named(pool, name, board),
        "models": by_model(stored),
        "spend": await pool.fetchval(SPENT, SCOPE, name) or 0.0,
        "gate": {"calls": gated["calls"], "refused": gated["refused"]},
        "lost": lost_of(stored),
        "records": dict(Counter(record.agent for record in records)),
    }


def cell(value: Any) -> str:
    if isinstance(value, list):
        text = ", ".join(cell(item) for item in value) if value else "none"
    elif isinstance(value, dict):
        text = json.dumps(value) if value else "none"
    elif isinstance(value, float):
        text = str(round(value, 6))
    elif value is None:
        text = "none"
    else:
        text = str(value)
    return text.replace("|", "\\|")


def flattened(key: str, value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict) and value:
        for inner, found in value.items():
            yield from flattened(f"{key}.{inner}", found)
    else:
        yield key, value


def tabled(value: Any) -> bool:
    return (
        isinstance(value, dict) and bool(value) and all(isinstance(v, dict) for v in value.values())
    )


def section(title: str, rows: dict[str, Any]) -> str:
    label = rows.get("label")
    if label not in LABELS:
        raise MeasureError(f"measure: a section's label is real, fake or arithmetic; got {label!r}")
    lines = [f"## {title}", "", f"`{label}`", ""]
    shown = {key: value for key, value in rows.items() if key != "label"}
    flat = [
        pair for key, value in shown.items() if not tabled(value) for pair in flattened(key, value)
    ]
    lines += ["| measure | value |", "|---|---|"]
    lines += [f"| {cell(key)} | {cell(value)} |" for key, value in flat]
    for key, value in shown.items():
        if not tabled(value):
            continue
        columns = list(dict.fromkeys(column for inner in value.values() for column in inner))
        lines += ["", f"| {cell(key)} | {' | '.join(map(cell, columns))} |"]
        lines.append("|---" * (len(columns) + 1) + "|")
        for name, inner in value.items():
            cells = " | ".join(cell(inner.get(column)) for column in columns)
            lines.append(f"| {cell(name)} | {cells} |")
    return "\n".join(lines) + "\n"


def append_results(path: Path, title: str, rows: dict[str, Any]) -> None:
    text = section(title, rows)
    held = path.read_text(encoding="utf-8") if path.exists() else ""
    gap = "" if not held or held.endswith("\n\n") else "\n" if held.endswith("\n") else "\n\n"
    with path.open("a", encoding="utf-8") as out:
        out.write(gap + text)


async def measured(url: str, name: str) -> dict[str, Any]:
    pool = await asyncpg.create_pool(url, server_settings={"search_path": "moon"})
    try:
        engine_pool = await asyncpg.create_pool(url, server_settings={"search_path": "moon_engine"})
        try:
            return await measure(pool, engine_pool, name)
        finally:
            await engine_pool.close()
    finally:
        await pool.close()


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print(USAGE_LINE, file=sys.stderr)
        return 2
    url = os.environ.get("RFNRY_EXAMPLES_POSTGRES_URL")
    if not url:
        print("measure: RFNRY_EXAMPLES_POSTGRES_URL is not set", file=sys.stderr)
        return 1
    name, path = argv[0], Path(argv[1])
    try:
        rows = asyncio.run(measured(url, name))
        append_results(path, argv[2] if len(argv) == 3 else name, rows)
    except (MeasureError, OSError, asyncpg.PostgresError) as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

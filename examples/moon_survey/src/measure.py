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
from rfnry_engine import Message, PostgresStorage, Text, ToolCall

import harness
import moon
from board import Board
from merge import cost_of, crosschecked

SCOPE = "moon"
LABELS = ("real", "fake", "arithmetic")
LEADS = ("lead", "flat", "captain", "boss")
UNKNOWN = "unknown"
USAGE = {"fake": "characters / 4, the fakes' estimate", "real": "as each provider reported it"}
REGION_ID = re.compile(r"[NS]\d\d-\d\d/\d+")
DIGITS = re.compile(r"(\d+)")
OURS = "(conversation = $2 or starts_with(conversation, $2 || '/'))"
SURVEY = "select mode, shape, status, regions from surveys where scope = $1 and survey = $2"
ROWS = f"""
select conversation, prompt, at, body::text as body from messages
where scope = $1 and {OURS}
order by conversation, position
"""
MOVES = f"select conversation, moves from windows where scope = $1 and {OURS}"
WAITS = """
select count(*) as count, coalesce(sum(wait), 0) as waited from provider_log
where scope = $1 and survey = $2 and status = 429
"""
GATE = """
select count(*) as calls, count(*) filter (where allowed is false) as refused from gate_log
where scope = $1 and survey = $2
"""
SPENT = "select total from spend where scope = $1 and survey = $2"
TWICE = "select count(*) from landed where scope = $1 and survey = $2 and attempts > 1"
USAGE_LINE = "usage: uv run src/measure.py <survey> <results.md> [<title>]"

Span = tuple[datetime, datetime]


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


def data_of(message: Message) -> Any:
    try:
        return json.loads(text_of(message.content))
    except ValueError:
        return None


def natural(key: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in DIGITS.split(key)]


def answered(rows: list[Stored], tools: tuple[str, ...]) -> list[tuple[ToolCall, Stored]]:
    asked: dict[str, ToolCall] = {}
    found = []
    for row in rows:
        message = row.message
        if message.role == "assistant":
            asked.update({call.id: call for call in message.tool_calls})
        elif message.role == "tool" and message.tool_call_id in asked:
            call = asked.pop(message.tool_call_id)
            if call.name in tools and not message.is_error:
                found.append((call, row))
    return found


def children(key: str, rows: list[Stored]) -> dict[str, str]:
    found = {}
    for call, row in answered(rows, ("Delegate",)):
        given = data_of(row.message)
        job = given.get("job") if isinstance(given, dict) else None
        agent = call.args.get("agent")
        if isinstance(job, str) and isinstance(agent, str):
            found[f"{key}/{job}"] = agent
    return found


def named(name: str, shape: str, stored: dict[str, list[Stored]]) -> dict[str, str]:
    found = {name: harness.TOP[shape]} if shape in harness.TOP else {}
    for key, rows in stored.items():
        if key.startswith(harness.boss_prefix(name)):
            found[key] = "boss"
        elif key.startswith(harness.surveyor_prefix(name)):
            found[key] = "surveyor"
        found.update(children(key, rows))
    return found


def begins(message: Message) -> bool:
    if message.role != "user":
        return False
    return message.notice is None or message.notice.kind == "lost"


def spans(rows: list[Stored]) -> list[Span]:
    found: list[Span] = []
    for row in rows:
        if not found or begins(row.message):
            found.append((row.at, row.at))
        else:
            found[-1] = (found[-1][0], row.at)
    return found


def seconds(found: Iterable[Span]) -> float:
    total, end = 0.0, None
    for first, last in sorted(found):
        if end is None or first > end:
            total += (last - first).total_seconds()
            end = last
        elif last > end:
            total += (last - end).total_seconds()
            end = last
    return round(total, 3)


def agents_of(
    stored: dict[str, list[Stored]], agents: dict[str, str], moves: dict[str, int]
) -> dict[str, dict[str, Any]]:
    table = {}
    for key in sorted(stored, key=natural):
        if agents.get(key) not in LEADS:
            continue
        rows = stored[key]
        prompt = [row.prompt for row in rows if row.prompt is not None]
        table[key] = {
            "agent": agents[key],
            "calls": len(prompt),
            "prompt": prompt,
            "wall_s": seconds(spans(rows)),
            "moves": moves.get(key, 0),
        }
    return table


def kinds_of(stored: dict[str, list[Stored]], agents: dict[str, str]) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    for key in sorted(stored, key=natural):
        prompt = [row.prompt for row in stored[key] if row.prompt is not None]
        entry = table.setdefault(
            agents.get(key, UNKNOWN), {"conversations": 0, "calls": 0, "largest_prompt": 0}
        )
        entry["conversations"] += 1
        entry["calls"] += len(prompt)
        entry["largest_prompt"] = max([entry["largest_prompt"], *prompt])
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


def jobs_of(stored: dict[str, list[Stored]]) -> dict[str, Any]:
    found = sorted(
        (row.at, len(text_of(row.message.content).encode()))
        for rows in stored.values()
        for _, row in answered(rows, ("Jobs",))
    )
    sizes = [size for _, size in found]
    return {"calls": len(sizes), "bytes": sizes, "max": max(sizes, default=0), "total": sum(sizes)}


def spread(counts: list[int]) -> dict[str, Any]:
    if not counts:
        return {"min": 0, "mean": 0.0, "max": 0}
    return {"min": min(counts), "mean": round(sum(counts) / len(counts), 2), "max": max(counts)}


def sharing(stored: dict[str, list[Stored]], agents: dict[str, str]) -> dict[str, Any]:
    written, read = [], []
    for key, rows in stored.items():
        if agents.get(key) != "surveyor":
            continue
        found = answered(rows, ("note", "notes"))
        written.append(sum(1 for call, _ in found if call.name == "note"))
        given = [data_of(row.message) for call, row in found if call.name == "notes"]
        read.append(sum(len(data.get("notes", [])) for data in given if isinstance(data, dict)))
    return {
        "surveyors": len(written),
        "written": sum(written),
        "written_per_surveyor": spread(written),
        "read_with_rereads": sum(read),
        "read_with_rereads_per_surveyor": spread(read),
    }


def names_a_neighbour(region: str, dossier: dict[str, Any]) -> bool:
    neighbours = set(moon.region(region).neighbours)
    said = [str(dossier.get("notes") or "")]
    for value in dossier["values"].values():
        said += [source["source"] for source in value.get("sources", [])]
    return any(found in neighbours for text in said for found in REGION_ID.findall(text))


async def measure(pool: asyncpg.Pool, engine_pool: asyncpg.Pool, name: str) -> dict[str, Any]:
    survey = await pool.fetchrow(SURVEY, SCOPE, name)
    if survey is None:
        raise MeasureError(f"measure: survey '{name}' does not exist")
    label = "fake" if survey["mode"] == "fake" else "real"
    stored: dict[str, list[Stored]] = {}
    for row in await pool.fetch(ROWS, SCOPE, name):
        stored.setdefault(row["conversation"], []).append(
            Stored(row["prompt"], row["at"], Message.model_validate_json(row["body"]))
        )
    agents = named(name, survey["shape"], stored)
    times = [row.at for rows in stored.values() for row in rows]
    moves = {row["conversation"]: row["moves"] for row in await pool.fetch(MOVES, SCOPE, name)}
    records = await PostgresStorage(engine_pool).run_tree(SCOPE, tags={"survey": name})
    board = Board(pool, SCOPE, name, live=lambda _: False)
    counts = await board.counts()
    landed = await board.landed()
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
        "elapsed_s": round((max(times) - min(times)).total_seconds(), 3) if times else 0.0,
        "running_s": seconds(span for rows in stored.values() for span in spans(rows)),
        "agents": agents_of(stored, agents, moves),
        "kinds": kinds_of(stored, agents),
        "jobs": jobs_of(stored),
        "moves": sum(moves.values()),
        "rate_limits": {"429s": waits["count"], "waited_s": round(waits["waited"], 3)},
        "notes": sharing(stored, agents),
        "dossiers": {
            "landed": len(landed),
            "naming_a_neighbour": sum(
                1 for row in landed if names_a_neighbour(row["region"], row["dossier"])
            ),
            "crosschecked": sum(1 for row in landed if crosschecked(row)),
        },
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

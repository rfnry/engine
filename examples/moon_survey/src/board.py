import json
from collections.abc import Callable
from typing import Any

import asyncpg
from rfnry_engine import Tool, Tools, current_call

import moon

KINDS = ("finding", "summary", "agree", "dispute")
JSON_COLUMNS = ("dossier", "checks", "data_check", "truth", "disputes")

SURVEY = "select second, regions from surveys where scope = $1 and survey = $2"
LANDED = "select run_id from landed where scope = $1 and survey = $2 and region = $3"
HOLDER = """
select run_id from claims where scope = $1 and survey = $2 and region = $3 for update
"""
HELD = """
select claims.region from claims
where claims.scope = $1 and claims.survey = $2 and claims.run_id = $3 and claims.region <> $4
    and not exists (
        select 1 from landed
        where landed.scope = $1 and landed.survey = $2 and landed.region = claims.region
    )
limit 1
"""
LAND = """
insert into landed (scope, survey, region, run_id, provider, simulated,
    dossier, checks, data_check, truth, disputes)
values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
on conflict (scope, survey, region) do update set attempts = landed.attempts + 1
returning attempts
"""


RELEASE = "delete from claims where scope = $1 and survey = $2 and region = $3"
LOCK = """
select 1 from claims where scope = $1 and survey = $2 and region = $3 for update
"""


class BoardError(Exception):
    pass


def checked_region(region: str) -> moon.Region:
    try:
        return moon.region(region)
    except moon.RegionError:
        raise BoardError(f"board: region '{region}' does not exist") from None


async def create_survey(
    pool: asyncpg.Pool,
    scope: str,
    survey: str,
    cap: float,
    second: set[str],
    regions: list[str],
) -> None:
    await pool.execute(
        """
        insert into surveys (scope, survey, cap, second, regions) values ($1, $2, $3, $4, $5)
        on conflict (scope, survey) do update
        set cap = excluded.cap, second = excluded.second, regions = excluded.regions
        """,
        scope,
        survey,
        cap,
        json.dumps(sorted(second)),
        json.dumps(list(regions)),
    )


class Board:
    def __init__(
        self, pool: asyncpg.Pool, scope: str, survey: str, *, live: Callable[[str], bool]
    ) -> None:
        self.pool = pool
        self.scope = scope
        self.survey = survey
        self.live = live

    async def _survey(self) -> tuple[set[str], list[str]]:
        found = await self.pool.fetchrow(SURVEY, self.scope, self.survey)
        if found is None:
            raise BoardError(f"board: survey '{self.survey}' does not exist")
        return set(json.loads(found["second"])), json.loads(found["regions"])

    async def regions(self) -> list[str]:
        return (await self._survey())[1]

    async def second(self) -> set[str]:
        return (await self._survey())[0]

    async def claim(self, region: str, run_id: str) -> dict[str, Any]:
        await self._inside(region)
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "select pg_advisory_xact_lock(hashtext($1))",
                f"{self.scope}/{self.survey}/{run_id}",
            )
            landed = await conn.fetchval(LANDED, self.scope, self.survey, region)
            if landed is not None:
                return {"claimed": False, "by": landed}
            held = await conn.fetchval(HELD, self.scope, self.survey, run_id, region)
            if held is not None:
                raise BoardError(f"board: run '{run_id}' already holds region '{held}'")
            while True:
                taken = await conn.fetchval(
                    """
                    insert into claims (scope, survey, region, run_id) values ($1, $2, $3, $4)
                    on conflict (scope, survey, region) do nothing
                    returning run_id
                    """,
                    self.scope,
                    self.survey,
                    region,
                    run_id,
                )
                holder = taken or await conn.fetchval(HOLDER, self.scope, self.survey, region)
                if holder is None:
                    continue
                landed = await conn.fetchval(LANDED, self.scope, self.survey, region)
                if landed is not None:
                    if taken is not None:
                        await conn.execute(RELEASE, self.scope, self.survey, region)
                    return {"claimed": False, "by": landed}
                if holder == run_id:
                    return {"claimed": True, "by": run_id}
                if self.live(holder):
                    return {"claimed": False, "by": holder}
                moved = await conn.fetchval(
                    """
                    update claims set run_id = $4, at = now()
                    where scope = $1 and survey = $2 and region = $3
                    returning run_id
                    """,
                    self.scope,
                    self.survey,
                    region,
                    run_id,
                )
                if moved is not None:
                    return {"claimed": True, "by": run_id}

    async def _inside(self, region: str) -> None:
        checked_region(region)
        _, regions = await self._survey()
        if region not in regions:
            raise BoardError(f"board: region '{region}' is not in survey '{self.survey}'")

    async def claim_of(self, run_id: str) -> str | None:
        return await self.pool.fetchval(
            "select region from claims where scope = $1 and survey = $2 and run_id = $3"
            " order by at desc, region limit 1",
            self.scope,
            self.survey,
            run_id,
        )

    async def fail(self, region: str, run_id: str, reason: str) -> None:
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "insert into failures (scope, survey, region, run_id, reason)"
                " values ($1, $2, $3, $4, $5)",
                self.scope,
                self.survey,
                region,
                run_id,
                reason,
            )
            await conn.execute(
                """
                delete from claims
                where scope = $1 and survey = $2 and region = $3 and run_id = $4
                    and not exists (
                        select 1 from landed
                        where scope = $1 and survey = $2 and region = $3
                    )
                """,
                self.scope,
                self.survey,
                region,
                run_id,
            )

    async def note(self, region: str, author: str, kind: str, text: str) -> int:
        if kind not in KINDS:
            raise BoardError(f"board: kind must be one of {', '.join(KINDS)}; got '{kind}'")
        await self._inside(region)
        return await self.pool.fetchval(
            "insert into notes (scope, survey, region, author, kind, text)"
            " values ($1, $2, $3, $4, $5, $6) returning id",
            self.scope,
            self.survey,
            region,
            author,
            kind,
            text,
        )

    async def _notes(self, column: str, value: str) -> list[dict[str, Any]]:
        rows = await self.pool.fetch(
            f"select id, region, author, kind, text, at from notes"
            f" where scope = $1 and survey = $2 and {column} = $3 order by id",
            self.scope,
            self.survey,
            value,
        )
        return [dict(row) for row in rows]

    async def notes(self, region: str) -> list[dict[str, Any]]:
        checked_region(region)
        return await self._notes("region", region)

    async def notes_by(self, author: str) -> list[dict[str, Any]]:
        return await self._notes("author", author)

    async def land(self, row: dict[str, Any]) -> bool:
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.fetchval(LOCK, self.scope, self.survey, row["region"])
            attempts = await conn.fetchval(
                LAND,
                self.scope,
                self.survey,
                row["region"],
                row["run_id"],
                row["provider"],
                bool(row["simulated"]),
                *(json.dumps(row[name]) for name in JSON_COLUMNS),
            )
        return attempts == 1

    async def update_landed(self, region: str, **fields: Any) -> None:
        unknown = sorted(set(fields) - set(JSON_COLUMNS))
        if unknown or not fields:
            raise BoardError(
                f"board: update_landed takes {', '.join(JSON_COLUMNS)}; got "
                f"{', '.join(unknown) or 'nothing'}"
            )
        names = list(fields)
        sets = ", ".join(f"{name} = ${i}" for i, name in enumerate(names, 4))
        changed = await self.pool.fetchval(
            f"update landed set {sets} where scope = $1 and survey = $2 and region = $3"
            " returning 1",
            self.scope,
            self.survey,
            region,
            *(json.dumps(fields[name]) for name in names),
        )
        if changed is None:
            raise BoardError(f"board: region '{region}' has not landed")

    async def landed(self, region: str | None = None) -> list[dict[str, Any]]:
        rows = await self.pool.fetch(
            """
            select region, run_id, provider, simulated, dossier, checks, data_check, truth,
                disputes, attempts, at
            from landed
            where scope = $1 and survey = $2 and ($3::text is null or region = $3)
            order by at, region
            """,
            self.scope,
            self.survey,
            region,
        )
        found = []
        for row in rows:
            item = dict(row)
            for name in JSON_COLUMNS:
                item[name] = json.loads(item[name])
            found.append(item)
        return found

    async def counts(self) -> dict[str, int]:
        _, regions = await self._survey()
        scoped = (self.scope, self.survey, regions)
        landed = {
            r["region"]
            for r in await self.pool.fetch(
                "select region from landed where scope = $1 and survey = $2"
                " and region = any($3::text[])",
                *scoped,
            )
        }
        held = {
            r["region"]: r["run_id"]
            for r in await self.pool.fetch(
                "select region, run_id from claims where scope = $1 and survey = $2"
                " and region = any($3::text[])",
                *scoped,
            )
        }
        failed = {
            r["region"]
            for r in await self.pool.fetch(
                "select distinct region from failures where scope = $1 and survey = $2"
                " and region = any($3::text[])",
                *scoped,
            )
        }
        claimed = {r for r, run in held.items() if r not in landed and self.live(run)}
        broken = failed - landed - claimed
        return {
            "landed": len(landed),
            "failed": len(broken),
            "claimed": len(claimed),
            "free": len(regions) - len(landed) - len(claimed) - len(broken),
        }


def caller() -> str:
    call = current_call()
    if call is None:
        raise BoardError("board: no run is calling")
    return call.run_id


def board_tools(board: Board) -> Tools:
    async def claim(args: dict[str, Any]) -> dict[str, Any]:
        return await board.claim(args["region"], caller())

    async def note(args: dict[str, Any]) -> dict[str, Any]:
        return {"note": await board.note(args["region"], caller(), args["kind"], args["text"])}

    async def notes(args: dict[str, Any]) -> dict[str, Any]:
        caller()
        found = await board.notes(args["region"])
        return {"notes": [{**n, "at": n["at"].isoformat()} for n in found]}

    async def region(args: dict[str, Any]) -> dict[str, Any]:
        caller()
        found = checked_region(args["region"])
        second = await board.second()
        return {
            "region": found.id,
            "bounds": {
                "lat_min": found.lat_min,
                "lat_max": found.lat_max,
                "lon_min": found.lon_min,
                "lon_max": found.lon_max,
            },
            "centre": list(found.centre),
            "neighbours": list(found.neighbours),
            "crosscheck": "crosscheck_second" if found.id in second else "crosscheck",
        }

    named = {
        "type": "object",
        "properties": {"region": {"type": "string", "description": "A region id, such as R17"}},
        "required": ["region"],
    }
    return Tools(
        [
            Tool(
                "claim",
                "Take a region for yourself. Answers claimed true when it is yours, false when"
                " another surveyor holds it or it is done. You hold one region at a time.",
                body=claim,
                input_schema=named,
            ),
            Tool(
                "note",
                "Leave a note on a region: a finding, a summary, or an agree or dispute"
                " when you check another surveyor's work.",
                body=note,
                input_schema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "A region id"},
                        "kind": {"type": "string", "enum": list(KINDS)},
                        "text": {"type": "string", "description": "What you found or conclude"},
                    },
                    "required": ["region", "kind", "text"],
                },
            ),
            Tool(
                "notes",
                "Read the notes left on a region, oldest first, each with its author and kind.",
                body=notes,
                input_schema=named,
            ),
            Tool(
                "region",
                "Read a region's bounds, centre, neighbouring regions and the crosscheck"
                " agent that checks it.",
                body=region,
                input_schema=named,
            ),
        ]
    )

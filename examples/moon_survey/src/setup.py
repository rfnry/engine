import asyncio
import hashlib
import os
import sys
from collections.abc import Callable
from pathlib import Path

import asyncpg
import httpx

import moon

USER_AGENT = "moon-survey-example/0"
TIMEOUT = 120.0
TABLES = """
create table if not exists messages (
    scope text not null,
    conversation text not null,
    position integer not null,
    body json not null,
    prompt integer,
    at timestamptz not null default now(),
    primary key (scope, conversation, position)
);
create table if not exists windows (
    scope text not null,
    conversation text not null,
    start integer not null,
    moves integer not null,
    since integer not null,
    primary key (scope, conversation)
);
create table if not exists surveys (
    scope text not null,
    survey text not null,
    cap double precision not null,
    second text not null,
    regions text not null,
    mode text not null,
    shape text not null,
    crews integer not null,
    status text not null,
    primary key (scope, survey)
);
create table if not exists claims (
    scope text not null,
    survey text not null,
    region text not null,
    run_id text not null,
    at timestamptz not null default now(),
    primary key (scope, survey, region)
);
create index if not exists claims_by_run on claims (scope, survey, run_id);
create table if not exists notes (
    id bigserial primary key,
    scope text not null,
    survey text not null,
    region text not null,
    author text not null,
    kind text not null,
    text text not null,
    at timestamptz not null default now()
);
create index if not exists notes_by_region on notes (scope, survey, region, id);
create index if not exists notes_by_author on notes (scope, survey, author, id);
create table if not exists landed (
    scope text not null,
    survey text not null,
    region text not null,
    run_id text not null,
    provider text not null,
    simulated boolean not null,
    dossier text not null,
    checks text not null,
    data_check text not null,
    truth text not null,
    disputes text not null,
    attempts integer not null default 1,
    at timestamptz not null default now(),
    primary key (scope, survey, region)
);
create table if not exists failures (
    scope text not null,
    survey text not null,
    region text not null,
    run_id text not null,
    reason text not null,
    at timestamptz not null default now()
);
create index if not exists failures_by_region on failures (scope, survey, region);
create table if not exists fetches (
    scope text not null,
    survey text not null,
    run_id text not null,
    url text not null,
    status integer not null,
    bytes integer not null,
    saved text,
    cached boolean not null,
    at timestamptz not null default now()
);
create index if not exists fetches_by_run on fetches (scope, survey, run_id);
create table if not exists spend (
    scope text not null,
    survey text not null,
    total double precision not null default 0,
    primary key (scope, survey)
);
create table if not exists gate_log (
    id bigserial primary key,
    scope text not null,
    survey text not null,
    run_id text not null,
    agent text not null,
    tool text not null,
    allowed boolean,
    probability double precision,
    error text,
    at timestamptz not null default now()
);
create index if not exists gate_log_by_run on gate_log (scope, survey, run_id);
create table if not exists provider_log (
    id bigserial primary key,
    scope text not null,
    survey text,
    provider text not null,
    model text not null,
    status integer,
    wait double precision not null,
    headers json not null,
    at timestamptz not null default now()
);
"""


class SetupError(Exception):
    pass


def open_url(url: str) -> bytes:
    response = httpx.get(
        url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, follow_redirects=True
    )
    response.raise_for_status()
    return response.content


def digest_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(sources: Path, into: Path, opener: Callable[[str], bytes]) -> list[Path]:
    into.mkdir(parents=True, exist_ok=True)
    kept = []
    for source in moon.parse_sources(sources):
        target = into / source.name
        if not (target.is_file() and digest_of(target) == source.sha256):
            body = opener(source.url)
            digest = hashlib.sha256(body).hexdigest()
            if digest != source.sha256:
                raise SetupError(
                    f"setup: {source.name} has sha256 {digest}; SOURCES.md says {source.sha256}"
                )
            if len(body) != source.size:
                raise SetupError(
                    f"setup: {source.name} has {len(body)} bytes; SOURCES.md says {source.size}"
                )
            partial = target.with_name(f"{source.name}.part")
            try:
                partial.write_bytes(body)
                partial.replace(target)
            except BaseException:
                partial.unlink(missing_ok=True)
                raise
        kept.append(target)
    return kept


def quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


async def create_tables(conn: asyncpg.Connection, schema: str = "moon") -> None:
    async with conn.transaction():
        await conn.execute(f"create schema if not exists {quoted(schema)}")
        await conn.execute(f"set local search_path to {quoted(schema)}")
        await conn.execute(TABLES)


async def tables(url: str) -> None:
    conn = await asyncpg.connect(url)
    try:
        await create_tables(conn)
    finally:
        await conn.close()


def main(argv: list[str]) -> int:
    if argv not in (["data"], ["tables"]):
        print("usage: uv run src/setup.py data | tables", file=sys.stderr)
        return 2
    try:
        if argv == ["data"]:
            for path in download(moon.SOURCES, moon.data_dir(), open_url):
                print(f"{path.name} ok")
        else:
            url = os.environ.get("RFNRY_EXAMPLES_POSTGRES_URL")
            if not url:
                raise SetupError("setup: RFNRY_EXAMPLES_POSTGRES_URL is not set")
            asyncio.run(tables(url))
            print("tables ok")
    except (SetupError, httpx.HTTPError, OSError, asyncpg.PostgresError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

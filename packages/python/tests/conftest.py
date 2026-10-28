import asyncio
import os
import signal
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import asyncpg
import pytest
from database import SCHEMA, URL, Schemas

from rfnry_engine.refinery import migrations
from rfnry_engine.refinery.postgres import PostgresStorage

CONTAINER = "start the rfnry-engine-pg container (docker start rfnry-engine-pg)"


async def execute(sql: str) -> None:
    connection = await asyncpg.connect(URL, timeout=5)
    try:
        await connection.execute(sql)
    finally:
        await connection.close()


def pytest_sessionstart(session: pytest.Session) -> None:
    if not URL:
        raise pytest.UsageError(
            f"tests: RFNRY_TEST_POSTGRES_URL is not set; {CONTAINER} and set "
            "RFNRY_TEST_POSTGRES_URL=postgresql://harness:harness@127.0.0.1:55435/harness"
        )
    try:
        asyncio.run(execute(f"CREATE SCHEMA {SCHEMA}"))
    except (
        OSError,
        asyncpg.PostgresError,
        asyncpg.InterfaceError,
        ValueError,
        TimeoutError,
    ) as exc:
        raise pytest.UsageError(
            f"tests: cannot connect to RFNRY_TEST_POSTGRES_URL ({type(exc).__name__}: {exc}); {CONTAINER}"
        ) from None


def pytest_sessionfinish(session: pytest.Session) -> None:
    asyncio.run(execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))


@pytest.fixture(scope="session")
async def pool() -> AsyncIterator[Any]:
    pool = await asyncpg.create_pool(URL, min_size=1, server_settings={"search_path": SCHEMA})
    async with pool.acquire() as conn:
        await migrations.apply(conn, migrations.plan_create)
    yield pool
    await pool.close()


@pytest.fixture
async def schemas() -> AsyncIterator[Schemas]:
    held = Schemas()
    yield held
    await held.close()


@pytest.fixture
async def storage(pool: Any) -> PostgresStorage:
    await pool.execute("DELETE FROM engine_labels; DELETE FROM engine_runs")
    return PostgresStorage(pool)


def _workers_left() -> list[int]:
    found = []
    for task in Path("/proc/self/task").glob("*/children"):
        for child in task.read_text().split():
            try:
                command = Path(f"/proc/{child}/cmdline").read_bytes()
                state = Path(f"/proc/{child}/stat").read_text().rsplit(")", 1)[1].split()[0]
            except OSError:
                continue
            if b"workers/process.py" in command and state != "Z":
                found.append(int(child))
    return found


@pytest.fixture(autouse=True)
def _no_worker_left_behind() -> Iterator[None]:
    yield
    left = _workers_left()
    for pid in left:
        os.killpg(pid, signal.SIGKILL)
    assert left == [], "tests: a test left a worker process running; close its engine"

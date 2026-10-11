import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from rfnry_engine.refinery.postgres import PostgresStorage

URL = os.environ.get("RFNRY_TEST_POSTGRES_URL", "")
SCHEMA = f"engine_test_{uuid.uuid4().hex}"
CONTAINER = "start the rfnry-engine-pg container (docker start rfnry-engine-pg)"


async def execute(sql: str) -> None:
    connection = await asyncpg.connect(URL)
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
    except (OSError, asyncpg.PostgresError) as exc:
        raise pytest.UsageError(
            f"tests: cannot create a test schema at RFNRY_TEST_POSTGRES_URL ({exc}); {CONTAINER}"
        ) from exc


def pytest_sessionfinish(session: pytest.Session) -> None:
    asyncio.run(execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))


@pytest.fixture(scope="session")
async def pool() -> AsyncIterator[Any]:
    pool = await asyncpg.create_pool(URL, min_size=1, server_settings={"search_path": SCHEMA})
    await PostgresStorage(pool).initialize()
    yield pool
    await pool.close()


@pytest.fixture
async def storage(pool: Any) -> PostgresStorage:
    await pool.execute("DELETE FROM engine_labels; DELETE FROM engine_runs")
    return PostgresStorage(pool)

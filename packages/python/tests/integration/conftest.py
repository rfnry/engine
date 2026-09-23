import os
from collections.abc import AsyncIterator
from typing import Any

import pytest

from rfnry_engine.refinery.inmemory import InMemoryStorage
from rfnry_engine.refinery.postgres import PostgresStorage
from rfnry_engine.refinery.storage import Storage


@pytest.fixture(params=["inmemory", pytest.param("postgres", marks=pytest.mark.postgres)])
async def storage(request: Any) -> AsyncIterator[Storage]:
    if request.param == "inmemory":
        yield InMemoryStorage()
        return
    url = os.environ.get("RFNRY_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("RFNRY_TEST_POSTGRES_URL is not set")
    asyncpg = pytest.importorskip("asyncpg")
    pool = await asyncpg.create_pool(url)
    postgres = PostgresStorage(pool)
    await postgres.initialize()
    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE engine_runs, engine_labels")
    yield postgres
    await pool.close()

from typing import Any

import pytest

from integration.stores import TABLE, PostgresStore


@pytest.fixture
async def store(pool: Any) -> PostgresStore:
    await pool.execute(TABLE)
    await pool.execute("DELETE FROM store_messages")
    return PostgresStore(pool)

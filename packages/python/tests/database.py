import asyncio
import os
import uuid
from typing import Any

import asyncpg

URL = os.environ.get("RFNRY_TEST_POSTGRES_URL", "")
SCHEMA = f"engine_test_{uuid.uuid4().hex}"


def quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


async def admin(statement: str) -> None:
    connection = await asyncpg.connect(URL, timeout=5)
    try:
        await connection.execute(statement)
    finally:
        await connection.close()


class Schemas:
    def __init__(self) -> None:
        self.made: list[str] = []
        self.pools: list[Any] = []

    def name(self, suffix: str = "") -> str:
        name = f"{SCHEMA[:24]}_{uuid.uuid4().hex[:8]}{suffix}"
        self.made.append(name)
        return name

    async def create(self, suffix: str = "") -> str:
        name = self.name(suffix)
        await admin(f"CREATE SCHEMA {quoted(name)}")
        return name

    async def pool(self, name: str) -> Any:
        pool = await asyncpg.create_pool(
            URL, min_size=1, max_size=4, timeout=5, server_settings={"search_path": quoted(name)}
        )
        self.pools.append(pool)
        return pool

    async def close(self) -> None:
        for pool in self.pools:
            await asyncio.wait_for(pool.close(), 5)
        for name in self.made:
            await admin(f"DROP SCHEMA IF EXISTS {quoted(name)} CASCADE")

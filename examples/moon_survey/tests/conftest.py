import os
import secrets

import asyncpg
import pytest
import pytest_asyncio


@pytest_asyncio.fixture(scope="session")
async def pool():
    url = os.environ.get("RFNRY_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("RFNRY_TEST_POSTGRES_URL is not set")
    schema = f"moon_test_{secrets.token_hex(6)}"
    admin = await asyncpg.connect(url)
    await admin.execute(f"CREATE SCHEMA {schema}")
    made = await asyncpg.create_pool(url, server_settings={"search_path": schema})
    try:
        yield made
    finally:
        await made.close()
        await admin.execute(f"DROP SCHEMA {schema} CASCADE")
        await admin.close()

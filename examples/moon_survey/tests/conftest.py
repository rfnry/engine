import asyncio
import os
import secrets
import sys
from pathlib import Path

import asyncpg
import pytest
import pytest_asyncio

import setup


@pytest_asyncio.fixture(scope="session")
async def pool():
    url = os.environ.get("RFNRY_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("RFNRY_TEST_POSTGRES_URL is not set")
    schema = f"moon_test_{secrets.token_hex(6)}"
    admin = await asyncpg.connect(url)
    made = None
    try:
        await admin.execute(f"CREATE SCHEMA {schema}")
        await setup.create_tables(admin, schema)
        made = await asyncpg.create_pool(url, server_settings={"search_path": schema})
        yield made
    finally:
        try:
            if made is not None:
                await made.close()
            await admin.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        finally:
            await admin.close()


@pytest_asyncio.fixture(scope="session")
async def engine_pool():
    url = os.environ.get("RFNRY_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("RFNRY_TEST_POSTGRES_URL is not set")
    schema = f"moon_engine_test_{secrets.token_hex(6)}"
    cli = Path(sys.executable).parent / "rfnry-engine"
    admin = await asyncpg.connect(url)
    made = None
    try:
        created = await asyncio.create_subprocess_exec(
            str(cli),
            "create",
            "--schema",
            schema,
            "--url",
            url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err = await created.communicate()
        if created.returncode:
            raise RuntimeError(f"rfnry-engine create failed: {err.decode()}")
        made = await asyncpg.create_pool(url, server_settings={"search_path": schema})
        yield made
    finally:
        try:
            if made is not None:
                await made.close()
            await admin.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        finally:
            await admin.close()


GRS_URL = (
    "https://pds-geosciences.wustl.edu/lunar/lp-l-grs-5-elem-abundance-v1/lp_9001/data/"
    "lpgrs_high1_elem_abundance_2deg.tab"
)


@pytest.fixture
def dossier():
    return {
        "region": "N75-90/0",
        "terrain": "polar",
        "values": {
            "water_ice": {
                "low": 0.0,
                "high": 1.5,
                "confidence": "low",
                "sources": [
                    "https://pds-geosciences.wustl.edu/missions/lunarp/reduced/hydrogenlow.txt"
                ],
            },
            "TiO2": {"low": 0.2, "high": 1.0, "confidence": "medium", "sources": [GRS_URL]},
            "FeO": {"low": 4, "high": 7, "confidence": "medium", "sources": [GRS_URL]},
            "Al2O3": {
                "low": 24,
                "high": 30,
                "confidence": "medium",
                "sources": ["Lunar Prospector gamma-ray spectrometer"],
            },
            "Th": {"low": 0.5, "high": 1.5, "confidence": "high", "sources": [GRS_URL]},
            "He3": {"unknown": "no orbiter measures He3 and no paper gives this region a value"},
        },
        "notes": "Cold traps near the pole may hold several wt% of ice.",
        "simulated": False,
    }

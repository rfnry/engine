import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg
import uvicorn
from fastapi import FastAPI

import harness
from routes import router

POOL_SIZE = 20


def settings() -> harness.Settings:
    return harness.Settings(
        work=Path(os.environ.get("MOON_SURVEY_WORK", str(harness.WORK))),
        scale=float(os.environ.get("MOON_SURVEY_FAKE_SCALE", "1")),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    url = os.environ.get("RFNRY_EXAMPLES_POSTGRES_URL")
    if not url:
        raise harness.SurveyError("survey: RFNRY_EXAMPLES_POSTGRES_URL is not set", 503)
    app_pool = await asyncpg.create_pool(
        url, max_size=POOL_SIZE, server_settings={"search_path": "moon"}
    )
    try:
        engine_pool = await asyncpg.create_pool(url, server_settings={"search_path": "moon_engine"})
        try:
            await harness.open(app_pool, engine_pool, settings())
            try:
                yield
            finally:
                await harness.close()
        finally:
            await engine_pool.close()
    finally:
        await app_pool.close()


app = FastAPI(lifespan=lifespan)
app.include_router(router)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=8000)

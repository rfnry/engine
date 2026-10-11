import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg
import uvicorn
from fastapi import FastAPI

import harness
from routes import router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with asyncpg.create_pool(os.environ["RFNRY_EXAMPLES_POSTGRES_URL"]) as pool:
        await harness.setup(pool)
        yield


app = FastAPI(lifespan=lifespan)
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

import asyncio
import os

import asyncpg

from harness import absent, ask, build, forget, read_back, review
from providers import AnthropicProvider

DSN = os.environ.get("ENGINE_POSTGRES_URL", "postgresql://engine:engine@127.0.0.1:55435/engine")


async def main() -> None:
    pool = await asyncpg.create_pool(DSN)
    refinery, engine = await build(pool, AnthropicProvider())

    run = await ask(engine, "What is the boiling point of water at sea level?")
    await review(refinery, run)
    print(run.reply)

    stored, labels = await read_back(refinery, run)
    print(f"stored {stored.agent} run at {stored.at:%H:%M:%S}", end="")
    print(f" with {len(labels)} label from {labels[0].source}")

    print(f"cleaned up {await forget(refinery)} rows", end="")
    print(f" · absent: {await absent(refinery)}")
    await pool.close()


asyncio.run(main())

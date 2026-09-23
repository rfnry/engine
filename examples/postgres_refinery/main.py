import asyncio
import os

import asyncpg
from rfnry_engine import Engine, PostgresStorage, Refinery, Review

from providers import AnthropicProvider

DSN = os.environ.get("ENGINE_POSTGRES_URL", "postgresql://harness:harness@127.0.0.1:55435/harness")


async def main() -> None:
    pool = await asyncpg.create_pool(DSN)
    refinery = Refinery(PostgresStorage(pool))
    await refinery.initialize()
    engine = Engine("agents", generation=AnthropicProvider(), refinery=refinery)

    run = await engine.run(
        "assistant", scope="postgres-demo", input="What is the boiling point of water at sea level?"
    )
    await refinery.review(
        "postgres-demo", run.run_id, Review(stance="approval", comment="Correct and short.")
    )
    print(run.reply)

    stored = await refinery.storage.get_run("postgres-demo", run.run_id)
    labels = await refinery.storage.labels("postgres-demo", run_ids=[run.run_id])
    print(f"stored {stored.agent} run at {stored.at:%H:%M:%S}", end="")
    print(f" with {len(labels)} label from {labels[0].source}")

    print(f"cleaned up {await refinery.delete_scope('postgres-demo')} rows", end="")
    print(f" · absent: {await refinery.verify_absent('postgres-demo')}")
    await pool.close()


asyncio.run(main())

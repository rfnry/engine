import asyncpg
from rfnry_engine import (
    Engine,
    GenerationModel,
    LabelRow,
    PostgresStorage,
    Refinery,
    Review,
    Run,
    RunRow,
)


async def build(pool: asyncpg.Pool, generation: GenerationModel) -> tuple[Refinery, Engine]:
    refinery = Refinery(PostgresStorage(pool))
    await refinery.initialize()
    return refinery, Engine("agents", generation=generation, refinery=refinery)


async def ask(engine: Engine, question: str) -> Run:
    return await engine.run("assistant", scope="postgres-demo", input=question)


async def review(refinery: Refinery, run: Run) -> None:
    await refinery.review(
        "postgres-demo", run.run_id, Review(stance="approval", comment="Correct and short.")
    )


async def read_back(refinery: Refinery, run: Run) -> tuple[RunRow | None, list[LabelRow]]:
    stored = await refinery.storage.get_run("postgres-demo", run.run_id)
    labels = await refinery.storage.labels("postgres-demo", run_ids=[run.run_id])
    return stored, labels


async def forget(refinery: Refinery) -> int:
    return await refinery.delete_scope("postgres-demo")


async def absent(refinery: Refinery) -> bool:
    return await refinery.verify_absent("postgres-demo")

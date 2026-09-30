import asyncio

from rfnry_engine import Engine, GenerationModel, Run, Tool


async def inventory(args):
    await asyncio.sleep(5)
    return {"sku": args["sku"], "units": 12}


def build(generation: GenerationModel) -> Engine:
    return Engine(
        "agents",
        generation=generation,
        tools=[Tool("inventory", body=inventory)],
        tool_timeout=1.0,
    )


async def ask(engine: Engine, question: str) -> Run:
    return await engine.run("stock", scope="demo", input=question)

import asyncio

from rfnry_engine import Engine, Tool

from providers import AnthropicProvider


async def inventory(args):
    await asyncio.sleep(5)
    return {"sku": args["sku"], "units": 12}


async def main() -> None:
    engine = Engine(
        "agents",
        generation=AnthropicProvider(),
        tools=[Tool("inventory", body=inventory)],
        tool_timeout=1.0,
    )

    run = await engine.run("stock", scope="demo", input="How many units of SKU-42 do we have?")
    for step in run.steps:
        print(f"{step.tool} {step.args} {step.duration_ms} ms error={step.is_error}: {step.result}")
    print(run.reply)


asyncio.run(main())

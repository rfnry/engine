import asyncio

from harness import ask, build
from providers import AnthropicProvider


async def main() -> None:
    engine = build(AnthropicProvider())

    run = await ask(engine, "How many units of SKU-42 do we have?")
    for step in run.steps:
        print(f"{step.tool} {step.args} {step.duration_ms} ms error={step.is_error}: {step.result}")
    print(run.reply)


asyncio.run(main())

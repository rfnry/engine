import asyncio

from rfnry_engine import InputError

from harness import build, describe, inspect
from providers import AnthropicProvider, GoogleProvider, TypeSafeProvider


async def main() -> None:
    for model in (AnthropicProvider(), GoogleProvider()):
        engine = build(model, TypeSafeProvider())
        run = await describe(engine)
        print(f"{run.model} · {run.usage.input} in / {run.usage.output} out")
        print(f"  {run.reply}\n")

    try:
        await inspect(engine)
    except InputError as error:
        print(f"inspector refused: {error}")


asyncio.run(main())

import asyncio

from harness import ask, build
from providers import AnthropicProvider


async def main() -> None:
    engine = build(AnthropicProvider())

    run = await ask(engine, "Why is the sky blue?")
    print(run.reply)
    usage = run.usage
    print(f"{run.model} · {run.calls} call · {usage.input} in / {usage.output} out", end="")
    print(f" · {run.duration_ms} ms\n")

    run = await ask(engine, "And why is it red at sunset?", history=run.conversation)
    print(run.reply)
    print(f"{len(run.messages)} messages sent · {run.usage.input} in / {run.usage.output} out")


asyncio.run(main())

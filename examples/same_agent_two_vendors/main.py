import asyncio

from harness import ask, build
from providers import AnthropicProvider, GoogleProvider

QUESTION = (
    "I pay for a €30 order at 16:00 on Friday. "
    "When does it reach Lisbon, and what does shipping cost?"
)


async def main() -> None:
    for model in (AnthropicProvider(), GoogleProvider()):
        engine = build(model)
        run = await ask(engine, QUESTION)
        print(f"{run.model}")
        print(f"  tools: {[f'{s.tool} {s.args}' for s in run.steps]}")
        usage = run.usage
        print(f"  {run.calls} calls · {usage.input} in / {usage.output} out", end="")
        print(f" · {run.duration_ms} ms")
        print(f"  {run.reply}\n")


asyncio.run(main())

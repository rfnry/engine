import asyncio

from rfnry_engine import TextDelta, ToolCallStarted

from harness import build, decide, explain
from providers import AnthropicProvider


async def show(event):
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, ToolCallStarted):
        print(f"\n[{event.agent} calls {event.tool}]")


async def main() -> None:
    engine = build(AnthropicProvider())
    request = "Order A-1 for $40 shows up twice on my card statement."

    await explain(engine, request, show)
    print("\n")

    verdict = await decide(engine, request, show)
    print(f"\nrefund={verdict.refund} amount={verdict.amount} reason={verdict.reason!r}")


asyncio.run(main())

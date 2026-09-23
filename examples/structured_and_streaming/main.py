import asyncio

from pydantic import BaseModel
from rfnry_engine import Engine, TextDelta, ToolCallStarted

from providers import AnthropicProvider


class Verdict(BaseModel):
    refund: bool
    amount: float
    reason: str


async def show(event):
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, ToolCallStarted):
        print(f"\n[{event.agent} calls {event.tool}]")


async def main() -> None:
    engine = Engine("agents", generation=AnthropicProvider())
    request = "Order A-1 for $40 shows up twice on my card statement."

    await engine.run(
        "reviewer",
        scope="demo",
        input=f"Explain your decision to the customer: {request}",
        on_event=show,
    )
    print("\n")

    run = await engine.run("reviewer", scope="demo", input=request, expect=Verdict, on_event=show)
    verdict = Verdict.model_validate_json(run.reply)
    print(f"\nrefund={verdict.refund} amount={verdict.amount} reason={verdict.reason!r}")


asyncio.run(main())

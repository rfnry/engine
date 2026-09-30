import asyncio

from harness import build, triage
from providers import AnthropicProvider


async def main() -> None:
    engine = build(AnthropicProvider())

    for message in [
        "Order A-1 was charged twice. What happens now?",
        "I want my $450 back for order B-7, I changed my mind.",
        "Order C-3 took forever to arrive. Not happy.",
        "Order A-1 has not shipped yet. What would express cost, and when would it arrive?",
        "I finish work at 22:00. Which pickup lockers are still open after that?",
    ]:
        run = await triage(engine, message)
        print(f"> {message}")
        for step in run.steps:
            error = " error" if step.is_error else ""
            print(f"  {step.tool}({step.args}) {step.duration_ms} ms{error}")
        print(f"{run.reply}\n")


asyncio.run(main())

import asyncio
from datetime import UTC, datetime

from rfnry_engine import Engine, Tool

from providers import AnthropicProvider

ORDERS = {
    "A-1": {
        "customer": "Dana",
        "amount": 40.0,
        "card": "visa 4242",
        "charges": ["2026-09-20 $40.00", "2026-09-20 $40.00"],
        "promised": "2026-09-24",
        "delivered": None,
    },
    "B-7": {
        "customer": "Sam",
        "amount": 450.0,
        "card": "amex 1005",
        "charges": ["2026-09-18 $450.00"],
        "promised": "2026-09-19",
        "delivered": "2026-09-19",
    },
    "C-3": {
        "customer": "Ines",
        "amount": 65.0,
        "card": "visa 7781",
        "charges": ["2026-09-10 $65.00"],
        "promised": "2026-09-15",
        "delivered": "2026-09-21",
    },
}


async def lookup_order(args):
    return ORDERS.get(args["order_id"], {"error": "no such order"})


async def now(args):
    return datetime.now(UTC).isoformat()


async def main() -> None:
    engine = Engine(
        "agents",
        generation=AnthropicProvider(),
        tools=[
            Tool("lookup_order", body=lookup_order),
            Tool("now", "The current UTC time", body=now),
        ],
    )

    for message in [
        "Order A-1 was charged twice. What happens now?",
        "I want my $450 back for order B-7, I changed my mind.",
        "Order C-3 took forever to arrive. Not happy.",
        "Order A-1 has not shipped yet. What would express cost, and when would it arrive?",
        "I finish work at 22:00. Which pickup lockers could I still collect a parcel from after that?",
    ]:
        run = await engine.run("support", scope="acme", input=message, task="triage")
        print(f"> {message}")
        for step in run.steps:
            error = " error" if step.is_error else ""
            print(f"  {step.tool}({step.args}) {step.duration_ms} ms{error}")
        print(f"{run.reply}\n")


asyncio.run(main())

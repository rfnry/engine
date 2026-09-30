from datetime import UTC, datetime

from rfnry_engine import Engine, GenerationModel, Run, Tool

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


def build(generation: GenerationModel) -> Engine:
    return Engine(
        "agents",
        generation=generation,
        tools=[
            Tool("lookup_order", body=lookup_order),
            Tool("now", "The current UTC time", body=now),
        ],
    )


async def triage(engine: Engine, message: str) -> Run:
    return await engine.run("support", scope="acme", input=message, task="triage")

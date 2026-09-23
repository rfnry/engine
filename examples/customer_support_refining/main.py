import asyncio
from pathlib import Path

from rfnry_engine import Engine, InMemoryStorage, Outcome, Refinery, Review, Tool, is_holdout

from providers import AnthropicProvider, TypeSafeProvider


async def lookup_order(args):
    return {"order_id": args["order_id"], "amount": 40.0, "charges": 2}


class Folder:
    def __init__(self, root):
        self.root = Path(root)

    async def put(self, key, data):
        (self.root / key).parent.mkdir(parents=True, exist_ok=True)
        (self.root / key).write_bytes(data)


async def main() -> None:
    refinery = Refinery(InMemoryStorage())
    engine = Engine(
        "agents",
        generation=AnthropicProvider(),
        decision=TypeSafeProvider(),
        refinery=refinery,
        tools=[Tool("lookup_order", body=lookup_order)],
    )

    refund = await engine.run(
        "support", "acme", "Order A-1 was charged twice. Can I get my money back?"
    )
    late = await engine.run("support", "acme", "Order A-2 still has not shipped. Where is it?")
    triage = await engine.run("triage", "acme", {"message": "Checkout is down for every customer."})

    await refinery.review(
        "acme",
        refund.run_id,
        Review(
            stance="approval",
            correction="Yes: the second $40 charge is refunded in full within five business days.",
        ),
    )
    await refinery.review(
        "acme", refund.run_id, Review(correction="Refund the duplicate $40 charge today.")
    )
    await refinery.review("acme", triage.run_id, Review(correction={"urgent": True, "severity": 3}))
    await refinery.outcome(
        "acme", late.run_id, Outcome(status="failed", detail="Customer escalated.")
    )

    labeled = await refinery.label("acme", engine, judge="judge", teacher="teacher")
    print(f"judged {labeled.judged} · taught {labeled.taught} · errors {labeled.errors}")
    for conflict in await refinery.conflicts("acme"):
        print(f"conflict on {conflict.run_id}: {[text for _, _, text in conflict.corrections]}")

    exported = await refinery.export("acme", Folder("datasets"))
    print(f"rows {exported.rows} · watermark {exported.watermark}")
    for key in exported.keys:
        print(f"  datasets/{key}")
    print(f"refund run is holdout: {is_holdout(refund.run_id)}")


asyncio.run(main())

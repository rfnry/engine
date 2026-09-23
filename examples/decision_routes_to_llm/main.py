import asyncio

from rfnry_engine import Engine

from providers import AnthropicProvider, TypeSafeProvider

MESSAGES = [
    "My invoice shows two charges for the same order.",
    "The export button does nothing since this morning.",
    "Do you have an office in Lisbon?",
]


async def main() -> None:
    engine = Engine("agents", generation=AnthropicProvider(), decision=TypeSafeProvider())

    for message in MESSAGES:
        run = await engine.run("frontdesk", scope="demo", input=message)
        intent = run.answers["intent"]
        print(f"> {message}\n  intent {intent.choice} (confidence {intent.confidence:.2f})")
        if run.routed_to:
            print(f"  routed to {run.routed_to.agent}: {run.routed_to.reply}\n")
        else:
            print("  not routed — your code decides what happens next\n")


asyncio.run(main())

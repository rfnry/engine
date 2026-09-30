import asyncio

from harness import build, receive
from providers import AnthropicProvider, TypeSafeProvider

MESSAGES = [
    "My invoice shows two charges for the same order.",
    "The export button does nothing since this morning.",
    "Do you have an office in Lisbon?",
]


async def main() -> None:
    engine = build(AnthropicProvider(), TypeSafeProvider())

    for message in MESSAGES:
        run = await receive(engine, message)
        intent = run.answers["intent"]
        print(f"> {message}\n  intent {intent.choice} (confidence {intent.confidence:.2f})")
        if run.routed_to:
            print(f"  routed to {run.routed_to.agent}: {run.routed_to.reply}\n")
        else:
            print("  not routed — your code decides what happens next\n")


asyncio.run(main())

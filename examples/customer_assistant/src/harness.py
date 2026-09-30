import asyncio
from collections.abc import AsyncIterator
from datetime import date
from pathlib import Path

from rfnry_engine import Engine, Event, Message, TextDelta, Tool

from providers import AnthropicProvider
from tools import lookup_order

ROOT = Path(__file__).resolve().parent.parent

engine = Engine(
    ROOT / "agents",
    generation=AnthropicProvider(),
    tools=[Tool("lookup_order", body=lookup_order)],
)
sessions: dict[str, list[Message]] = {}


async def ask(question: str, session: str | None) -> dict:
    run = await answer(question, session)
    return {"reply": run.reply, "run_id": run.run_id}


async def ask_stream(question: str, session: str | None) -> AsyncIterator[dict]:
    deltas: asyncio.Queue[str | None] = asyncio.Queue()

    async def on_event(event: Event) -> None:
        if isinstance(event, TextDelta):
            deltas.put_nowait(event.text)

    work = asyncio.create_task(answer(question, session, on_event))
    work.add_done_callback(lambda _: deltas.put_nowait(None))
    while (text := await deltas.get()) is not None:
        yield {"text": text}
    run = await work
    yield {"done": True, "run_id": run.run_id}


async def answer(question: str, session: str | None, on_event=None):
    history = sessions.get(session, []) if session else []
    run = await engine.run(
        "support",
        scope="acme-home",
        input=question,
        history=history,
        context=f"Today is {date.today().isoformat()}.",
        on_event=on_event,
    )
    if session:
        sessions[session] = run.conversation
    return run

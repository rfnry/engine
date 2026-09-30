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


async def run(question: str, session: str | None, on_event=None) -> dict:
    history = sessions.get(session, []) if session else []
    engine_run = await engine.run(
        "support",
        scope="acme-home",
        input=question,
        history=history,
        context=f"Today is {date.today().isoformat()}.",
        on_event=on_event,
    )
    if session:
        sessions[session] = engine_run.conversation
    return {"reply": engine_run.reply, "run_id": engine_run.run_id}


async def run_stream(question: str, session: str | None) -> AsyncIterator[dict]:
    deltas: asyncio.Queue[str | None] = asyncio.Queue()

    async def on_event(event: Event) -> None:
        if isinstance(event, TextDelta):
            deltas.put_nowait(event.text)

    work = asyncio.create_task(run(question, session, on_event))
    work.add_done_callback(lambda _: deltas.put_nowait(None))
    while (text := await deltas.get()) is not None:
        yield {"text": text}
    result = await work
    yield {"done": True, "run_id": result["run_id"]}

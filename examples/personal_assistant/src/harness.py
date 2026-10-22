from datetime import date
from pathlib import Path

from rfnry_engine import Document, Engine, Image, Message, Text

from providers import AnthropicProvider

ROOT = Path(__file__).resolve().parent.parent

engine = Engine(ROOT / "agents", models={"strong": AnthropicProvider()})
sessions: dict[str, tuple[list[Message], int]] = {}


def window(conversation: list[Message], start: int, turns: int = 20, budget: int = 150_000) -> int:
    starts = [i for i, m in enumerate(conversation) if m.role == "user" and i >= start]
    used = next((m.usage for m in reversed(conversation) if m.usage), None)
    if used and used.input + used.cache_read + used.cache_creation > budget and len(starts) > 1:
        return starts[len(starts) // 2]
    return starts[-turns] if len(starts) >= 2 * turns else start


async def run(question: str, session: str | None, attached: Image | Document | None) -> dict:
    conversation, start = sessions.get(session, ([], 0)) if session else ([], 0)
    start = window(conversation, start)
    parts = [Text(text=question), *([attached] if attached else [])]
    engine_run = await engine.run(
        "assistant",
        scope="jordan",
        input=parts,
        history=conversation,
        start=start,
        context=f"Today is {date.today().isoformat()}.",
    )
    if session:
        sessions[session] = (conversation + engine_run.added, start)
    return {"reply": engine_run.reply, "run_id": engine_run.run_id}

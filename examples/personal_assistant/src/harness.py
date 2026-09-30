from datetime import date
from pathlib import Path

from rfnry_engine import Document, Engine, Image, Message, Text

from providers import AnthropicProvider

ROOT = Path(__file__).resolve().parent.parent

engine = Engine(ROOT / "agents", generation=AnthropicProvider())
sessions: dict[str, list[Message]] = {}


async def run(question: str, session: str | None, attached: Image | Document | None) -> dict:
    history = sessions.get(session, []) if session else []
    parts = [Text(text=question), *([attached] if attached else [])]
    engine_run = await engine.run(
        "assistant",
        scope="jordan",
        input=parts,
        history=history,
        context=f"Today is {date.today().isoformat()}.",
    )
    if session:
        sessions[session] = engine_run.conversation
    return {"reply": engine_run.reply, "run_id": engine_run.run_id}

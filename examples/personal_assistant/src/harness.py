from datetime import date
from pathlib import Path

from rfnry_engine import Document, Engine, Image, Message, Text

from providers import AnthropicProvider

ROOT = Path(__file__).resolve().parent.parent

engine = Engine(ROOT / "agents", generation=AnthropicProvider())
sessions: dict[str, list[Message]] = {}


def attachment(media_type: str, name: str, data: bytes) -> Image | Document:
    if media_type.startswith("image/"):
        return Image(media_type=media_type, data=data)
    return Document(media_type=media_type, name=name, data=data)


async def ask(question: str, session: str | None, attached: Image | Document | None) -> dict:
    history = sessions.get(session, []) if session else []
    parts = [Text(text=question), *([attached] if attached else [])]
    run = await engine.run(
        "assistant",
        scope="jordan",
        input=parts,
        history=history,
        context=f"Today is {date.today().isoformat()}.",
    )
    if session:
        sessions[session] = run.conversation
    return {"reply": run.reply, "run_id": run.run_id}

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import harness

router = APIRouter()


class Ask(BaseModel):
    question: str
    session: str | None = None


@router.post("/ask")
async def ask(body: Ask) -> dict:
    return await harness.ask(body.question, body.session)


@router.post("/ask/stream")
async def ask_stream(body: Ask) -> StreamingResponse:
    chunks = harness.ask_stream(body.question, body.session)
    events = (f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n" async for chunk in chunks)
    return StreamingResponse(events, media_type="text/event-stream")

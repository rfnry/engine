import json

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import harness


class Ask(BaseModel):
    question: str
    session: str | None = None


app = FastAPI()


@app.post("/ask")
async def ask(body: Ask) -> dict:
    return await harness.ask(body.question, body.session)


@app.post("/ask/stream")
async def ask_stream(body: Ask) -> StreamingResponse:
    chunks = harness.ask_stream(body.question, body.session)
    events = (f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n" async for chunk in chunks)
    return StreamingResponse(events, media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

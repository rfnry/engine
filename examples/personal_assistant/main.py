from typing import Annotated

import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile

import harness

app = FastAPI()


@app.post("/ask")
async def ask(
    question: Annotated[str, Form()],
    session: Annotated[str | None, Form()] = None,
    file: UploadFile | None = None,
) -> dict:
    attached = None
    if file:
        if not (file.content_type or "").startswith(("image/", "application/pdf")):
            raise HTTPException(415, "send a photo or a PDF")
        attached = harness.attachment(file.content_type, file.filename or "", await file.read())
    return await harness.ask(question, session, attached)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

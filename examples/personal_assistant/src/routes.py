from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, UploadFile

import harness

router = APIRouter()


@router.post("/ask")
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

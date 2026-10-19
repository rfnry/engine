from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, WebSocket
from pydantic import BaseModel
from rfnry_engine import NotFoundError, Review

import chat
import harness

router = APIRouter()
Tenant = Annotated[str, Header(alias="X-Tenant")]


class Ask(BaseModel):
    contract: str
    question: str


@router.post("/ask")
async def ask(body: Ask, firm: Tenant) -> dict:
    if body.contract not in harness.CONTRACTS:
        raise HTTPException(404, f"there is no contract {body.contract}")
    return await harness.run(firm, body.contract, body.question)


@router.websocket("/chat")
async def talk(socket: WebSocket, firm: str, contract: str) -> None:
    if contract not in harness.CONTRACTS:
        await socket.close(code=4404, reason=f"there is no contract {contract}")
        return
    await socket.accept()
    await chat.serve(socket, firm, contract)


@router.post("/runs/{run_id}/review")
async def review(run_id: str, body: Review, firm: Tenant) -> dict:
    try:
        return await harness.review(firm, run_id, body)
    except NotFoundError as error:
        raise HTTPException(404, str(error)) from None


@router.post("/export")
async def export(firm: Tenant) -> dict:
    return await harness.export(firm)

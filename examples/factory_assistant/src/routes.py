from typing import Annotated

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

import harness

router = APIRouter()
Tenant = Annotated[str, Header(alias="X-Tenant")]


class Ask(BaseModel):
    question: str


@router.post("/ask")
async def ask(body: Ask, plant: Tenant) -> dict:
    if plant not in harness.PLANTS:
        raise HTTPException(404, f"there is no plant {plant}")
    return await harness.ask(plant, body.question)

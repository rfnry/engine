from typing import Annotated

from fastapi import APIRouter, Header
from pydantic import BaseModel

import harness

router = APIRouter()
Tenant = Annotated[str, Header(alias="X-Tenant")]


class Intake(BaseModel):
    message: str


@router.post("/intake")
async def intake(body: Intake, clinic: Tenant) -> dict:
    return await harness.intake(clinic, body.message)

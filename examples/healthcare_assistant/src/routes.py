from typing import Annotated

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

import harness

router = APIRouter()
Tenant = Annotated[str, Header(alias="X-Tenant")]


class Intake(BaseModel):
    message: str


@router.post("/intake")
async def intake(body: Intake, clinic: Tenant) -> dict:
    if clinic not in harness.CLINICS:
        raise HTTPException(404, f"there is no clinic {clinic}")
    return await harness.run(clinic, body.message)

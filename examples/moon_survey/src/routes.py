from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import harness

router = APIRouter()


class Started(BaseModel):
    name: str
    mode: Literal["fake", "real", "hostile"]
    shape: Literal["crews", "flat", "app"]
    size: int = 250
    regions: list[str] | None = None
    cap: float
    crews: int | None = None


class Resumed(BaseModel):
    cap: float | None = None


def refused(error: harness.SurveyError) -> HTTPException:
    return HTTPException(status_code=error.status, detail=str(error))


@router.post("/surveys", status_code=202)
async def start(body: Started) -> dict[str, Any]:
    try:
        return await harness.run(
            body.name,
            mode=body.mode,
            shape=body.shape,
            size=body.size,
            regions=body.regions,
            cap=body.cap,
            crews=body.crews,
        )
    except harness.SurveyError as error:
        raise refused(error) from None


@router.get("/surveys/{name}")
async def read(name: str) -> dict[str, Any]:
    try:
        return await harness.progress(name)
    except harness.SurveyError as error:
        raise refused(error) from None


@router.post("/surveys/{name}/resume", status_code=202)
async def resume(name: str, body: Resumed | None = None) -> dict[str, Any]:
    try:
        return await harness.resume(name, cap=None if body is None else body.cap)
    except harness.SurveyError as error:
        raise refused(error) from None


@router.get("/map")
async def shown(survey: str | None = None) -> dict[str, Any]:
    try:
        return await harness.map(survey)
    except harness.SurveyError as error:
        raise refused(error) from None

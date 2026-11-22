import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request

import harness

router = APIRouter()

STARTED = ("name", "mode", "shape", "size", "regions", "cap", "crews")
RESUMED = ("cap",)


def refused(error: harness.SurveyError) -> HTTPException:
    return HTTPException(status_code=error.status, detail=str(error))


async def body_of(request: Request, fields: tuple[str, ...]) -> dict[str, Any]:
    raw = await request.body()
    if not raw.strip():
        return {}
    try:
        body = json.loads(raw)
    except ValueError:
        raise HTTPException(400, "survey: the body must be a JSON object") from None
    if not isinstance(body, dict):
        raise HTTPException(400, "survey: the body must be a JSON object")
    unknown = sorted(set(body) - set(fields))
    if unknown:
        raise HTTPException(
            400,
            f"survey: the body takes {', '.join(fields)}; got {', '.join(map(str, unknown))}",
        )
    return body


@router.post("/surveys", status_code=202)
async def start(request: Request) -> dict[str, Any]:
    body = await body_of(request, STARTED)
    try:
        return await harness.run(
            body.get("name"),
            mode=body.get("mode"),
            shape=body.get("shape"),
            size=body.get("size", 250),
            regions=body.get("regions"),
            cap=body.get("cap"),
            crews=body.get("crews"),
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
async def resume(name: str, request: Request) -> dict[str, Any]:
    body = await body_of(request, RESUMED)
    try:
        return await harness.resume(name, cap=body.get("cap"))
    except harness.SurveyError as error:
        raise refused(error) from None


@router.get("/map")
async def shown(survey: str | None = None) -> dict[str, Any]:
    try:
        return await harness.map(survey)
    except harness.SurveyError as error:
        raise refused(error) from None

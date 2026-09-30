from typing import Annotated

import uvicorn
from fastapi import FastAPI, Header
from pydantic import BaseModel

import harness

Tenant = Annotated[str, Header(alias="X-Tenant")]


class Intake(BaseModel):
    message: str


app = FastAPI()


@app.post("/intake")
async def intake(body: Intake, clinic: Tenant) -> dict:
    return await harness.intake(clinic, body.message)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

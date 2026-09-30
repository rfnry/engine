from typing import Annotated, Literal

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

import harness

Tenant = Annotated[str, Header(alias="X-Tenant")]


class Ask(BaseModel):
    question: str
    task: Literal["troubleshoot", "procedure"] | None = None


app = FastAPI()


@app.post("/ask")
async def ask(body: Ask, plant: Tenant) -> dict:
    if plant not in harness.PLANTS:
        raise HTTPException(404, f"there is no plant {plant}")
    return await harness.ask(plant, body.question, body.task)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

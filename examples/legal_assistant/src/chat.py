import asyncio
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from rfnry_engine import Event, Message, MessagesTaken, Stop

import harness

AGENT = "reviewer"
OFFERED = {"task": ["review"], "skill": ["redline", "risk"]}


def menu() -> dict[str, dict[str, str]]:
    agent = harness.engine.agents[AGENT]
    have = {"task": agent.tasks, "skill": agent.skills}
    return {
        f"{kind}s": {name: have[kind][name] for name in names} for kind, names in OFFERED.items()
    }


def command(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("/"):
        return {}, text
    kind, _, rest = text.removeprefix("/").partition(" ")
    if kind not in OFFERED:
        commands = " and ".join(f"/{kind}" for kind in OFFERED)
        raise ValueError(f"/{kind} is not a command; the commands are {commands}")
    name, _, rest = rest.strip().partition(" ")
    if name not in OFFERED[kind]:
        raise ValueError(f"/{kind} takes one of {', '.join(OFFERED[kind])}")
    return {kind: name}, rest


class Chat:
    def __init__(self, socket: WebSocket, firm: str, contract: str) -> None:
        self.socket = socket
        self.firm = firm
        self.contract = contract
        self.conversation: list[Message] = []
        self.inbox: asyncio.Queue[str] = asyncio.Queue()
        self.stop = Stop()
        self.work: asyncio.Task[None] | None = None
        self.sending = asyncio.Lock()
        self.open = True

    async def send(self, kind: str, **body: Any) -> None:
        async with self.sending:
            if self.open:
                await self.socket.send_json({"type": kind, **body})

    async def show(self, event: Event) -> None:
        await self.send(type(event).__name__, **event.model_dump(mode="json"))
        if isinstance(event, MessagesTaken):
            await self.send("read", count=len(event.messages))

    async def receive(self, said: dict[str, Any]) -> None:
        text = str(said.get("text", ""))
        if said.get("type") == "stop":
            self.stop.now()
        elif self.work is not None and not self.work.done():
            if text.startswith("/"):
                await self.send("refused", text=text, reason="a command waits for the reply")
            else:
                self.inbox.put_nowait(text)
                await self.send("queued", text=text)
        else:
            try:
                invoked, text = command(text)
            except ValueError as error:
                await self.send("refused", text=text, reason=str(error))
                return
            self.work = asyncio.create_task(self.answer(text, invoked))

    async def answer(self, text: str, invoked: dict[str, str]) -> None:
        while True:
            self.stop = Stop()
            run = await harness.engine.run(
                AGENT,
                scope=self.firm,
                input=text,
                history=self.conversation,
                context=harness.CONTRACTS[self.contract],
                on_event=self.show,
                stop=self.stop,
                inbox=self.inbox,
                **invoked,
            )
            self.conversation = run.conversation
            idle = self.inbox.empty()
            if idle:
                self.work = None
            await self.send("done", run_id=run.run_id, stopped=run.stopped, reply=run.reply)
            if idle:
                return
            text, invoked = self.inbox.get_nowait(), {}
            await self.send("read", count=1)

    async def close(self) -> None:
        self.open = False
        self.stop.now()
        if self.work is not None:
            await asyncio.gather(self.work, return_exceptions=True)


async def serve(socket: WebSocket, firm: str, contract: str) -> None:
    chat = Chat(socket, firm, contract)
    await chat.send("menu", **menu())
    try:
        while True:
            await chat.receive(await socket.receive_json())
    except WebSocketDisconnect:
        await chat.close()

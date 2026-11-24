from collections.abc import Sequence

from rfnry_engine import chain
from rfnry_engine.conversation import Conversation
from rfnry_engine.models import Message, Pause, _handed, _open


class Transcript:
    head: bytes
    chained: chain.Chain
    begun: int
    sent: list[Message]
    seen: list[Message]
    settled: bool
    open_mark: Pause | None
    open_at: int

    def __init__(self, conversation: Conversation | None) -> None:
        self.conversation = conversation
        self.messages: list[Message] = []
        self.window = 0

    def start(
        self,
        history: list[Message],
        repairs: list[Message],
        fresh: Sequence[Message],
        resuming: Pause | None,
        head: bytes,
        chained: chain.Chain,
    ) -> None:
        self.head, self.chained = head, chained
        self.messages.extend([*history, *repairs])
        if resuming is None:
            self.messages.extend(fresh)
        self.begun = 0 if self.conversation is None else len(history) + len(repairs)
        if resuming is not None and self.conversation is not None:
            self.begun = next(
                (n for n in reversed(range(len(history))) if history[n].tool_calls),
                len(history) - 1,
            )
        self.sent = self.messages[self.begun : len(history) if resuming is not None else None]
        self.seen = self.messages
        self.settled = False
        self.open_mark, self.open_at = resuming, len(history) - 1

    @property
    def key(self) -> str | None:
        return None if self.conversation is None else self.conversation.key

    @property
    def lost(self) -> BaseException | None:
        return None if self.conversation is None else self.conversation.failed

    async def save(self) -> None:
        if self.conversation is not None:
            await self.conversation.save(self.messages)

    async def position(self) -> int | None:
        if self.conversation is None:
            return None
        return await self.conversation.window(len(self.messages))

    async def windowed(self, at: int | None) -> None:
        if at is None:
            if not self.settled:
                self.sent, self.settled = list(self.messages), True
            return
        if self.settled and at == self.window:
            self.seen.extend(self.messages[at + len(self.seen) :])
            return
        self.chained = chain.Chain(self.head)
        self.seen = await self.chained.settle(self.messages[at:])
        self.window = at
        if not self.settled:
            self.begun, self.sent, self.settled = at, list(self.seen), True

    def sending(self, preload: Sequence[Message]) -> list[Message]:
        return [
            _handed(message, sent=True)
            for message in (*preload, *self.seen)
            if message.pause is None
        ]

    async def stamp(self) -> str:
        return await self.chained.stamp(self.seen)

    def after_mark(self) -> list[Message]:
        return [] if self.open_mark is None else self.messages[self.open_at + 1 :]

    def past_mark(self) -> bool:
        return len(self.messages) > self.open_at + 1

    def mark(self, pause: Pause) -> None:
        self.messages.append(Message(role="assistant", content="[paused]", pause=pause))
        self.open_mark, self.open_at = pause, len(self.messages) - 1

    def remark(self, pause: Pause) -> None:
        if _open(pause) or pause.held:
            self.mark(pause)

    def close(self, pause: Pause) -> None:
        if _open(pause):
            self.mark(pause)
        else:
            self.messages.extend(pause.held)

    def forget_mark(self) -> None:
        self.open_mark = None

    def for_record(self) -> tuple[int, list[Message], list[Message]]:
        return (
            self.begun,
            self.sent,
            [*self.sent, *self.messages[self.begun + len(self.sent) :]],
        )

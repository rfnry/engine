from collections.abc import Awaitable, Callable, Collection, Sequence

from pydantic import TypeAdapter, ValidationError

from rfnry_engine.errors import InputError
from rfnry_engine.models import ConversationStore, Message, Pause, _handed, _open

HISTORY = TypeAdapter(list[Message])


def live(history: Sequence[Message]) -> Pause | None:
    pause = history[-1].pause if history else None
    return pause if pause is not None and _open(pause) else None


def open_calls(pause: Pause | None) -> frozenset[str]:
    if pause is None:
        return frozenset()
    asking = () if pause.asking is None else (pause.asking,)
    return frozenset([*(waiting.call.id for waiting in pause.waiting), *pause.delegated, *asking])


def ended_calls(history: Sequence[Message], named: Collection[str] = ()) -> list[Message]:
    last = next(
        (
            n
            for n in reversed(range(len(history)))
            if history[n].role == "assistant" and history[n].tool_calls
        ),
        None,
    )
    if last is None:
        return []
    answered = {message.tool_call_id for message in history[last + 1 :] if message.role == "tool"}
    return [
        Message(
            role="tool",
            tool_call_id=call.id,
            content=f"Ended: {call.name} did not return; the run that made the call ended, and what it did is unknown",
            is_error=True,
        )
        for call in history[last].tool_calls
        if call.id not in answered and call.id not in named
    ]


def _measured(key: str, length: object) -> int:
    if not isinstance(length, int) or isinstance(length, bool) or length < 0:
        raise InputError(
            f"conversation: the store's length of conversation {key!r} must be a whole number from 0; got {length!r}"
        )
    return length


class Conversation:
    def __init__(self, store: ConversationStore, scope: str, key: str) -> None:
        self.store = store
        self.scope = scope
        self.key = key
        self.saved = 0
        self.failed: BaseException | None = None

    async def _called[T](self, method: Callable[..., Awaitable[T]], key: str, *args: object) -> T:
        if self.failed is not None:
            raise self.failed
        try:
            return await method(self.scope, key, *args)
        except Exception as exc:
            self.failed = exc
            raise

    async def opened(self) -> list[Message]:
        length = _measured(self.key, await self._called(self.store.length, self.key))
        if length == 0:
            return []
        found = await self._called(self.store.read, self.key, 0)
        try:
            held = HISTORY.validate_python(found, from_attributes=True)
        except ValidationError as exc:
            raise InputError(
                f"conversation: the store's conversation {self.key!r} does not fit the contract: "
                f"{exc.errors(include_url=False, include_input=False)}"
            ) from exc
        if len(held) != length:
            raise InputError(
                f"conversation: the store holds {length} messages of conversation {self.key!r} but read returned {len(held)}"
            )
        self.saved = length
        return [_handed(message) for message in held]

    async def used(self, key: str) -> bool:
        length = await self._called(self.store.length, key)
        try:
            return _measured(key, length) > 0
        except InputError as exc:
            self.failed = exc
            raise

    async def window(self, length: int) -> int:
        start = await self._called(self.store.window, self.key, length)
        if not isinstance(start, int) or isinstance(start, bool) or not 0 <= start < length:
            raise InputError(
                f"conversation: the store's window for conversation {self.key!r} must be a position from 0 to {length - 1}; got {start!r}"
            )
        return start

    async def save(self, held: Sequence[Message]) -> None:
        end = len(held)
        if self.failed is not None or end <= self.saved:
            return
        fresh = [_handed(message) for message in held[self.saved : end]]
        await self._called(self.store.append, self.key, self.saved, fresh)
        self.saved = end

from collections.abc import Awaitable, Callable, Sequence

from pydantic import TypeAdapter, ValidationError

from rfnry_engine.errors import InputError
from rfnry_engine.models import ConversationStore, Message, _handed

HISTORY = TypeAdapter(list[Message])


class Conversation:
    def __init__(self, store: ConversationStore, scope: str, key: str) -> None:
        self.store = store
        self.scope = scope
        self.key = key
        self.saved = 0
        self.failed: BaseException | None = None

    async def _called[T](self, method: Callable[..., Awaitable[T]], *args: object) -> T:
        try:
            return await method(self.scope, self.key, *args)
        except BaseException as exc:
            self.failed = exc
            raise

    async def opened(self) -> list[Message]:
        length = await self._called(self.store.length)
        if not isinstance(length, int) or isinstance(length, bool) or length < 0:
            raise InputError(
                f"conversation: the store's length of conversation {self.key!r} must be a whole number from 0; got {length!r}"
            )
        if length == 0:
            return []
        found = await self._called(self.store.read, 0)
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

    async def save(self, held: Sequence[Message]) -> None:
        end = len(held)
        if self.failed is not None or end <= self.saved:
            return
        fresh = [_handed(message) for message in held[self.saved : end]]
        await self._called(self.store.append, fresh)
        self.saved = end

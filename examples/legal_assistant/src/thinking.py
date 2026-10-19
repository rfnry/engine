from typing import Any

from rfnry_engine import Message, OnDelta, Reply, ToolSpec

from providers import AnthropicProvider


class ThinkingProvider(AnthropicProvider):
    def _request(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> dict[str, Any]:
        return {
            **super()._request(system, messages, tools),
            "thinking": {"type": "adaptive", "display": "summarized"},
        }

    async def generate_stream(
        self, system: str, messages: list[Message], tools: list[ToolSpec], on_delta: OnDelta
    ) -> Reply:
        async with self.client.messages.stream(**self._request(system, messages, tools)) as stream:
            async for event in stream:
                if event.type == "thinking":
                    await on_delta(event.thinking, reasoning=True)
                elif event.type == "text":
                    await on_delta(event.text)
            return self._reply(await stream.get_final_message())

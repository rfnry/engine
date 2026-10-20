from rfnry_engine.chain import opening, preload_digest
from rfnry_engine.models import Text, ToolSpec


def test_the_opening_digest_is_stable_so_a_conversation_keeps_its_reasoning() -> None:
    tool = ToolSpec(name="Read", description="Read a file", input_schema={"type": "object"})
    digest = opening("model-a", "You are helpful.", [tool], preload_digest([Text(text="hello")]))
    assert digest.hex() == "fb2bbac4b13cc7076457d52bb61c187a119f02d50957ffa7459edf657a12e53c"

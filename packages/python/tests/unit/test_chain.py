from rfnry_engine.chain import opening
from rfnry_engine.models import Text, ToolSpec


def test_the_opening_digest_keeps_the_boot_key_so_a_conversation_keeps_its_reasoning() -> None:
    tool = ToolSpec(name="Read", description="Read a file", input_schema={"type": "object"})
    digest = opening("model-a", "You are helpful.", [tool], [Text(text="hello")])
    assert digest.hex() == "ba4b8057e87d816749f3944c929473d38827edf0604d5b2385dd28630b17fb7d"

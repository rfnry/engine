from rfnry_engine.chain import opening
from rfnry_engine.models import Text, ToolSpec


def test_the_opening_digest_is_stable_so_a_conversation_keeps_its_reasoning() -> None:
    tool = ToolSpec(name="Read", description="Read a file", input_schema={"type": "object"})
    digest = opening("model-a", "You are helpful.", [tool], [Text(text="hello")])
    assert digest.hex() == "94dc54d4369a0d0409cb572115a9152db3bf64bd52b9bd47237877a462b995fa"

import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from rfnry_engine import Engine
from rfnry_engine.errors import ConfigurationError
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.tools import Tool
from rfnry_engine.tree import load_tree

from .fakes import Scripted


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def agent_md(name: str, kind: str = "generation", model: str = "strong", extra: str = "") -> str:
    return f"---\nname: {name}\nkind: {kind}\nmodel: {model}\n{extra}---\nWork.\n"


class OrderArgs(BaseModel):
    """Which order to fetch."""

    order_id: str = Field(description="The order id, such as A-1001")


async def fetch(args: dict[str, Any]) -> Any:
    return {"order": args["order_id"]}


def shop(root: Path, tools: str = "[fetch_orders]") -> Path:
    write(root / "shop/AGENT.md", agent_md("shop", extra=f"tools: {tools}\n"))
    return root


def fetch_orders(description: str = "Fetch the customer's orders") -> Tool:
    return Tool("fetch_orders", description, body=fetch, args=OrderArgs)


async def test_an_agent_gets_the_tools_it_lists_with_their_code_descriptions(
    tmp_path: Path,
) -> None:
    engine = Engine(shop(tmp_path), models={}, tools=[fetch_orders()])
    defined = {d["name"]: d for d in EngineTools(engine, agent="shop", scope="acme").definitions()}
    assert defined["fetch_orders"]["description"] == "Fetch the customer's orders"
    assert defined["fetch_orders"]["input_schema"]["description"] == "Which order to fetch."
    assert (
        defined["fetch_orders"]["input_schema"]["properties"]["order_id"]["description"]
        == "The order id, such as A-1001"
    )
    assert engine.agents["shop"].tools == ["fetch_orders"]
    assert await EngineTools(engine, agent="shop", scope="acme").dispatch(
        "fetch_orders", {"order_id": "A-1"}
    ) == {"order": "A-1"}


@pytest.mark.parametrize(
    ("tools", "said"),
    [
        ("[' x']", "tools: an entry must not be blank or padded"),
        ("['']", "tools: an entry must not be blank or padded"),
        (
            "[fetch_orders, fetch_orders]",
            "tools: 'fetch_orders' is listed twice; list each tool once",
        ),
        ("[3]", "tools"),
    ],
)
def test_a_tools_list_that_does_not_fit_is_refused_naming_the_file(
    tmp_path: Path, tools: str, said: str
) -> None:
    with pytest.raises(ConfigurationError) as caught:
        load_tree(shop(tmp_path, tools))
    assert str(tmp_path / "shop/AGENT.md") in str(caught.value) and said in str(caught.value)


def test_a_decision_agent_cannot_list_tools(tmp_path: Path) -> None:
    write(
        tmp_path / "triage/AGENT.md",
        agent_md("triage", kind="decision", model="judge", extra="tools: [x]\n"),
    )
    with pytest.raises(ConfigurationError, match="decision agent 'triage' cannot list tools:"):
        load_tree(tmp_path)


@pytest.mark.parametrize(
    ("listed", "extra"),
    [
        ("Read", ""),
        ("History", ""),
        ("Delegate", "agents: [shop]\n"),
        ("Shell", "workspace: true\n"),
    ],
)
def test_listing_a_tool_the_engine_gives_is_refused(
    tmp_path: Path, listed: str, extra: str
) -> None:
    write(tmp_path / "shop/AGENT.md", agent_md("shop", extra=f"tools: [{listed}]\n{extra}"))
    with pytest.raises(
        ConfigurationError,
        match=f"lists '{listed}', a tool the engine gives it; drop it from tools:",
    ):
        Engine(tmp_path, models={}, tools=[Tool(listed, "Mine", body=fetch)])


def test_a_tool_named_delegate_is_listable_by_an_agent_without_agents(tmp_path: Path) -> None:
    engine = Engine(
        shop(tmp_path, "[Delegate]"), models={}, tools=[Tool("Delegate", "Mine", body=fetch)]
    )
    assert engine.agents["shop"].tools == ["Delegate"]


@pytest.mark.parametrize("models", [{}, {"strong": Scripted()}])
def test_a_tool_no_agent_lists_is_refused(tmp_path: Path, models: Any) -> None:
    with pytest.raises(ConfigurationError, match=re.escape("tools no agent lists: ['refund']")):
        Engine(
            shop(tmp_path),
            models=models,
            tools=[fetch_orders(), Tool("refund", "Refund a charge", body=fetch)],
        )


def test_a_listed_tool_without_a_tool_is_refused_when_the_agent_s_model_is_given(
    tmp_path: Path,
) -> None:
    Engine(shop(tmp_path), models={})
    with pytest.raises(ConfigurationError, match="lists tool 'fetch_orders' but no Tool was given"):
        Engine(shop(tmp_path), models={"strong": Scripted()})


@pytest.mark.parametrize(
    ("tool", "said"),
    [
        (Tool("fetch_orders", "Fetch"), "tool 'fetch_orders' has no body"),
        (Tool("fetch_orders", "", body=fetch), "tool 'fetch_orders' has no description"),
        (Tool("fetch_orders", "   ", body=fetch), "tool 'fetch_orders' has no description"),
        (Tool("fetch_orders", None, body=fetch), "tool 'fetch_orders' has no description"),  # type: ignore[arg-type]
    ],
)
def test_a_catalog_tool_needs_a_body_and_a_description(
    tmp_path: Path, tool: Tool, said: str
) -> None:
    with pytest.raises(ConfigurationError, match=said):
        Engine(shop(tmp_path), models={}, tools=[tool])


def test_two_agents_listing_one_tool_both_receive_it(tmp_path: Path) -> None:
    write(tmp_path / "lead/AGENT.md", agent_md("lead", extra="tools: [fetch_orders]\n"))
    write(tmp_path / "aide/AGENT.md", agent_md("aide", extra="tools: [fetch_orders]\n"))
    engine = Engine(tmp_path, models={}, tools=[fetch_orders()])
    for agent in ("lead", "aide"):
        names = [d["name"] for d in EngineTools(engine, agent=agent, scope="acme").definitions()]
        assert "fetch_orders" in names


def test_a_tool_only_an_agent_without_its_model_lists_loads_and_fails_when_it_runs(
    tmp_path: Path,
) -> None:
    write(tmp_path / "lead/AGENT.md", agent_md("lead", extra="agents: [writer]\n"))
    write(
        tmp_path / "writer/AGENT.md",
        agent_md("writer", model="fast", extra="tools: [fetch_orders]\n"),
    )
    engine = Engine(tmp_path, models={"strong": Scripted()}, tools=[fetch_orders()])
    assert engine.agents["writer"].tools == ["fetch_orders"]
    bare = Engine(tmp_path, models={"strong": Scripted()})
    with pytest.raises(ConfigurationError, match="lists tool 'fetch_orders' but no Tool was given"):
        EngineTools(bare, agent="writer", scope="acme")


def test_a_leftover_tools_md_and_tools_folder_are_ordinary_content(tmp_path: Path) -> None:
    write(tmp_path / "shop/AGENT.md", agent_md("shop", extra="preload: true\n"))
    write(tmp_path / "shop/TOOLS.md", "---\nfetch: Fetch\n---\n")
    write(tmp_path / "shop/tools/fetch.md", "---\ndescription: Fetch\n---\nHow to fetch.\n")
    spec = load_tree(tmp_path)["shop"]
    assert spec.tools == []
    assert "TOOLS.md" in spec.listing and "tools/" in spec.listing and "fetch.md" in spec.listing
    sent = "".join(str(part) for part in spec.preload)
    assert "How to fetch." in sent and "fetch: Fetch" in sent


def test_the_tools_an_agent_lists_are_part_of_its_fingerprint(tmp_path: Path) -> None:
    before = load_tree(shop(tmp_path, "[fetch_orders]"))["shop"].fingerprint
    after = load_tree(shop(tmp_path, "[fetch_orders, refund]"))["shop"].fingerprint
    assert before != after


def test_every_tool_without_a_body_is_refused_before_any_without_a_description(
    tmp_path: Path,
) -> None:
    blank = Tool("fetch_orders", "", body=fetch)
    bodiless = Tool("refund", "Refund a charge")
    with pytest.raises(ConfigurationError, match="tool 'refund' has no body"):
        Engine(shop(tmp_path), models={}, tools=[blank, bodiless])

from pathlib import Path
from typing import Any

from rfnry_engine import Engine
from rfnry_engine.engine import OrchestrationInfo, RouteInfo

from .trees import add_agent, provided, solo

ROUTING = {"by": "go", "to": {"next": "full"}}
QUESTION = (
    '{"type": "choice", "instructions": "Where?", '
    '"options": {"stay": "Keep it", "next": "Pass it on"}}'
)
FULL: dict[str, Any] = {
    "kind": "generation",
    "model": "strong",
    "instructions": "full.md",
    "context": "own",
    "max_model_calls": 7,
    "workspace": True,
    "tools": {"timeout": 12.5, "max_parallel": 3},
    "orchestration": {
        "delegate_to": ["bare"],
        "max_depth": 4,
        "max_parallel": 2,
        "background": True,
        "max_background": 6,
    },
    "modes": {
        "quick": {
            "max_model_calls": 2,
            "workspace": False,
            "tools": {"timeout": 1.5, "max_parallel": 1},
            "orchestration": {"max_depth": 1, "max_parallel": 1, "max_background": 1},
        }
    },
}


def _folder(root: Path) -> Path:
    solo(root, name="bare")
    add_agent(root, "full", FULL, files={"full.md": "Hi.\n"})
    add_agent(
        root,
        "router",
        {
            "kind": "decision",
            "model": "judge",
            "instructions": "router.md",
            "questions": ["router/*.json"],
            "orchestration": {"route": ROUTING},
        },
        files={"router.md": "", "router/go.json": QUESTION},
    )
    return root


def _engine(root: Path) -> Engine:
    return Engine(
        _folder(root),
        providers=provided({"strong": None, "judge": None}),
        max_model_calls=99,
        max_depth=9,
        max_background=8,
    )


def test_agent_info_shows_every_setting_of_the_entry(tmp_path: Path) -> None:
    full = _engine(tmp_path).agents["full"]
    assert full.context == "own"
    assert full.max_model_calls == 7
    assert full.workspace is True
    assert full.tool_timeout == 12.5
    assert full.tool_max_parallel == 3
    assert full.delegate_to == ["bare"]
    assert full.orchestration == OrchestrationInfo(
        max_depth=4, max_parallel=2, route=None, background=True, max_background=6
    )


def test_agent_info_of_a_router_shows_its_route(tmp_path: Path) -> None:
    route = _engine(tmp_path).agents["router"].orchestration.route
    assert route == RouteInfo(by="go", to={"next": "full"}, min_confidence=None)


def test_agent_info_of_an_entry_that_writes_none_of_them_reads_the_defaults(
    tmp_path: Path,
) -> None:
    bare = _engine(tmp_path).agents["bare"]
    assert bare.context == "inherit"
    assert bare.max_model_calls is None and bare.tool_timeout is None
    assert bare.tool_max_parallel is None
    assert bare.workspace is False
    assert bare.orchestration == OrchestrationInfo(
        max_depth=None, max_parallel=None, route=None, background=False, max_background=None
    )


def test_agent_info_of_a_mode_shows_the_mode_s_overlaid_values(tmp_path: Path) -> None:
    quick = _engine(tmp_path).agents["full"].modes["quick"]
    assert quick.context == "own"
    assert quick.max_model_calls == 2
    assert quick.workspace is False
    assert quick.tool_timeout == 1.5 and quick.tool_max_parallel == 1
    assert quick.orchestration == OrchestrationInfo(
        max_depth=1, max_parallel=1, route=None, background=True, max_background=1
    )

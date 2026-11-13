import copy
import json
from pathlib import Path
from typing import Any

import jsonschema_rs
import pytest
import yaml

from rfnry_engine.cli import main
from rfnry_engine.engine import Engine, check_folder
from rfnry_engine.errors import ConfigurationError
from rfnry_engine.manifest import read_manifest

from .fakes import Answering, Scripted
from .trees import provided, write_folder

AGENT = {"kind": "generation", "model": "strong", "instructions": "a.md"}
MODES = {"chat": {"description": "Chat."}, "email": {"description": "Mail."}}


def _good(root: Path) -> Path:
    return write_folder(
        root,
        {
            "a": {**AGENT, "files": {"allow": ["notes/*.md"]}, "modes": MODES},
            "b": {"kind": "decision", "model": "judge", "instructions": "a.md"},
        },
        files={"a.md": "Work.\n", "notes/x.md": "x\n"},
    )


def test_check_passes_a_good_folder(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["folder", "check", str(_good(tmp_path))]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == ["a  generation  strong  modes: chat, email", "b  decision  judge"]


def test_check_lists_the_runtime_tools(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = write_folder(
        tmp_path,
        {
            "a": {
                **AGENT,
                "tools": {"allow": ["handoff"]},
                "modes": {"chat": {"tools": {"allow": ["verify"]}}},
            },
            "b": {"kind": "decision", "model": "judge", "instructions": "a.md"},
        },
        runtime_tools=["handoff", "verify"],
        files={"a.md": "Work.\n"},
    )
    assert main(["folder", "check", str(folder)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "a  generation  strong  modes: chat  runtime tools: handoff, verify",
        "b  decision  judge",
    ]
    info = check_folder(folder)["a"]
    assert (info.runtime_tools, info.modes["chat"].runtime_tools) == (
        ["handoff"],
        ["handoff", "verify"],
    )


def test_check_fails_a_bad_one_with_its_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_folder(tmp_path, {"a": {**AGENT, "instructions": "missing.md"}})
    assert main(["folder", "check", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "missing.md" in captured.err


@pytest.mark.parametrize(
    ("name", "said"),
    [
        ("rfnry.yaml", "is the file itself; pass the folder that holds it, {folder}"),
        ("a.md", "is a file, not a folder; pass the folder that holds rfnry.yaml"),
    ],
)
def test_check_given_a_file_says_to_pass_the_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], name: str, said: str
) -> None:
    folder = _good(tmp_path).resolve()
    assert main(["folder", "check", str(folder / name)]) == 1
    assert capsys.readouterr().err == (f"manifest: {folder / name} {said.format(folder=folder)}\n")


def test_check_applies_the_default_process_ceiling(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_folder(
        tmp_path,
        {"a": {**AGENT, "worker": "w"}},
        workers={"w": {"processes": 20}},
        files={"a.md": "Work.\n"},
    )
    assert main(["folder", "check", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "max_worker_processes=16" in err and "--max-worker-processes" in err
    assert "Engine(" not in err
    assert main(["folder", "check", str(tmp_path), "--max-worker-processes", "32"]) == 0
    assert "a  generation" in capsys.readouterr().out


@pytest.mark.parametrize("flag", [["--url", "postgresql://x"], ["--schema", "engine"]])
def test_folder_takes_neither_url_nor_schema(flag: list[str], tmp_path: Path) -> None:
    for words in (["check", str(tmp_path)], ["schema"]):
        with pytest.raises(SystemExit) as stopped:
            main(["folder", *words, *flag])
        assert stopped.value.code == 2


def test_a_failure_is_one_line_on_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_folder(tmp_path, {"a": {**AGENT, "instructions": "missing.md"}})
    assert main(["folder", "check", str(tmp_path)]) == 1
    assert len(capsys.readouterr().err.splitlines()) == 1


def test_check_starts_no_process_for_a_folder_with_a_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    started: list[object] = []
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: started.append(args))
    folder = write_folder(
        tmp_path,
        {"a": {**AGENT, "tools": {"allow": ["one"]}}},
        tools={"one": {"module": "tools/one.py", "description": "Does one"}},
        files={"a.md": "Work.\n", "tools/one.py": "def run(args, call):\n    return {}\n"},
    )
    assert main(["folder", "check", str(folder)]) == 0
    assert "a  generation" in capsys.readouterr().out
    assert started == []


def test_check_needs_no_database_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("RFNRY_TEST_POSTGRES_URL", "PGHOST", "PGUSER", "PGPASSWORD", "PGDATABASE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("rfnry_engine.cli.find_spec", lambda name: None)
    monkeypatch.chdir(_good(tmp_path))
    assert main(["folder", "check"]) == 0
    assert "a  generation" in capsys.readouterr().out


def test_schema_is_valid_json_schema_and_names_agents(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["folder", "schema"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert "agents" in schema["properties"] and "workers" in schema["properties"]
    assert "$defs" in schema
    jsonschema_rs.meta.validate(schema)
    assert "schema" in schema["$defs"]["ToolEntry"]["properties"]


DESIGN = """\
version: 1

workers:
  customers:
    processes: 2
    threads: 32
    max_result_bytes: 33554432
  live:
    processes: 1

models:
  strong:
    provider: anthropic
    model: claude-sonnet-5-5
    effort: medium
  decision: {provider: typesafe, model: jev-latest}

tools:
  lookup_order:
    module: tools/lookup_order.py
    function: run
    description: Finds an order
    schema: tools/lookup_order.json
    timeout: 20
  refund: {module: tools/refund.py, schema: tools/refund.json}

runtime_tools: [handoff, verify_customer]

agents:
  support:
    kind: generation
    model: strong
    description: Handles support
    instructions: support/support.md
    max_model_calls: 24
    worker: customers
    files:
      allow: [knowledge]
      deny: [knowledge/drafts/**]
      preload: [support/rules]
      gitignore: true
    tools:
      allow: ["*", handoff]
      deny: [refund]
      timeout: 30
      max_parallel: 4
    skills:
      allow: [support/skills/*]
      deny: [internal]
      preload: [support/skills/tone]
      public: [escalate]
    orchestration:
      delegate_to: [drafter]
      max_depth: 1
      max_parallel: 2
    workspace: true
    modes:
      email:
        skills:
          preload: [support/channels/email]
      chat:
        worker: live
        tools:
          allow: [verify_customer]
        skills:
          preload: [support/channels/live, support/channels/chat]

  triage:
    kind: decision
    model: decision
    modes:
      email: {instructions: triage/email.md, questions: [triage/email/*.json]}
      chat: {instructions: triage/chat.md, questions: [triage/chat/*.json]}
    orchestration:
      route: {by: category, to: {billing: support}, min_confidence: 0.7}
"""


def _schema(capsys: pytest.CaptureFixture[str]) -> jsonschema_rs.Validator:
    assert main(["folder", "schema"]) == 0
    return jsonschema_rs.validator_for(json.loads(capsys.readouterr().out))


def _at(data: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    changed = copy.deepcopy(data)
    *above, last = path.split(".")
    held = changed
    for key in above:
        held = held[key]
    held[last] = value
    return changed


def test_the_schema_accepts_the_design_example(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "rfnry.yaml").write_text(DESIGN, encoding="utf-8")
    assert sorted(read_manifest(tmp_path).agents) == ["support", "triage"]
    assert _schema(capsys).is_valid(yaml.safe_load(DESIGN))


def test_the_schema_takes_a_tool_as_a_path_or_an_object(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["folder", "schema"]) == 0
    schema = json.loads(capsys.readouterr().out)
    entry = schema["properties"]["tools"]["additionalProperties"]
    assert {branch.get("type") for branch in entry["anyOf"]} >= {"string"}
    assert "short" not in schema["$defs"]["ToolEntry"]["properties"]
    short = _at(yaml.safe_load(DESIGN), "tools.lookup_order", "tools/lookup_order.py")
    assert _schema(capsys).is_valid(short)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("agents.support.files", None),
        ("tools.lookup_order", ""),
        ("runtime_tools", []),
        ("runtime_tools", [""]),
        ("runtime_tools", ["handoff", "handoff"]),
        ("agents.support.files.allow", []),
        ("agents.support.files", {}),
        ("agents.support.tools.allow", [""]),
        ("agents.support.description", ""),
        ("agents.support.modes", {}),
        ("agents.support.modes.email", None),
        ("models.strong.model", None),
        ("models.strong.effort", ""),
        ("workers", {}),
        ("workers.live.processes", None),
        ("agents.triage.orchestration.route.to", {}),
    ],
)
def test_the_schema_refuses_what_the_loader_refuses(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], path: str, value: Any
) -> None:
    changed = _at(yaml.safe_load(DESIGN), path, value)
    (tmp_path / "rfnry.yaml").write_text(yaml.safe_dump(changed), encoding="utf-8")
    with pytest.raises(ConfigurationError):
        read_manifest(tmp_path)
    assert not _schema(capsys).is_valid(changed)


def test_agents_lists_modes(tmp_path: Path) -> None:
    engine = Engine(
        _good(tmp_path), providers=provided({"strong": Scripted(), "judge": Answering()})
    )
    info = engine.agents["a"]
    assert list(info.modes) == ["chat", "email"]
    assert info.modes["chat"].description == "Chat."
    assert info.modes["chat"].modes == {}
    assert info.modes["chat"].name == "a"
    assert (info.worker, info.delegate_to, info.files) == ("default", [], ["notes/x.md"])

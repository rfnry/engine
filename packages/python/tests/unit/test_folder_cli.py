import json
from pathlib import Path

import jsonschema_rs
import pytest

from rfnry_engine.cli import main
from rfnry_engine.engine import Engine

from .fakes import Answering, Scripted
from .trees import write_folder

AGENT = {"kind": "generation", "model": "strong", "instructions": "a.md"}
MODES = {"chat": {"description": "Chat."}, "email": {"description": "Mail."}}


def _good(root: Path) -> Path:
    return write_folder(
        root,
        {
            "a": {**AGENT, "files": {"include": ["notes/*.md"]}, "modes": MODES},
            "b": {"kind": "decision", "model": "judge", "instructions": "a.md"},
        },
        files={"a.md": "Work.\n", "notes/x.md": "x\n"},
    )


def test_check_passes_a_good_folder(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["folder", "check", str(_good(tmp_path))]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == ["a  generation  strong  modes: chat, email", "b  decision  judge"]


def test_check_fails_a_bad_one_with_its_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_folder(tmp_path, {"a": {**AGENT, "instructions": "missing.md"}})
    assert main(["folder", "check", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "missing.md" in captured.err


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
        {"a": {**AGENT, "tools": {"include": ["one"]}}},
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


def test_agents_lists_modes(tmp_path: Path) -> None:
    engine = Engine(_good(tmp_path), models={"strong": Scripted(), "judge": Answering()})
    info = engine.agents["a"]
    assert list(info.modes) == ["chat", "email"]
    assert info.modes["chat"].description == "Chat."
    assert info.modes["chat"].modes == {}
    assert info.modes["chat"].name == "a"
    assert (info.worker, info.delegate_to, info.files) == ("default", [], ["notes/x.md"])

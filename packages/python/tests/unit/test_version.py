import tomllib
from pathlib import Path

import rfnry_engine

PACKAGE = Path(__file__).parents[2]


def test_the_package_the_project_and_the_lock_carry_one_version() -> None:
    project = tomllib.loads((PACKAGE / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    lock = tomllib.loads((PACKAGE / "uv.lock").read_text(encoding="utf-8"))
    locked = [p["version"] for p in lock["package"] if p["name"] == project["name"]]
    assert {"package": rfnry_engine.__version__, "lock": locked} == {
        "package": project["version"],
        "lock": [project["version"]],
    }

import hashlib
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

import moon

SOURCES = Path(__file__).resolve().parent.parent / "data" / "lp" / "SOURCES.md"
USER_AGENT = "moon-survey-example/0"
TIMEOUT = 120.0


class SetupError(Exception):
    pass


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    size: int
    sha256: str


def parse_sources(path: Path) -> list[Source]:
    found = []
    for line in path.read_text().splitlines():
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(cells) >= 4 and cells[1].startswith("https://"):
            found.append(Source(cells[0], cells[1], int(cells[2]), cells[3]))
    return found


def open_url(url: str) -> bytes:
    response = httpx.get(
        url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, follow_redirects=True
    )
    response.raise_for_status()
    return response.content


def download(sources: Path, into: Path, opener: Callable[[str], bytes]) -> list[Path]:
    into.mkdir(parents=True, exist_ok=True)
    kept = []
    for source in parse_sources(sources):
        target = into / source.name
        if not target.exists():
            body = opener(source.url)
            digest = hashlib.sha256(body).hexdigest()
            if digest != source.sha256:
                raise SetupError(
                    f"setup: {source.name} has sha256 {digest}; SOURCES.md says {source.sha256}"
                )
            target.write_bytes(body)
        kept.append(target)
    return kept


def main(argv: list[str]) -> int:
    if argv != ["data"]:
        print("usage: uv run src/setup.py data", file=sys.stderr)
        return 2
    try:
        for path in download(SOURCES, moon.data_dir(), open_url):
            print(f"{path.name} ok")
    except (SetupError, httpx.HTTPError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

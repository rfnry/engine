from pathlib import Path


class Folder:
    def __init__(self, root: Path) -> None:
        self.root = root

    async def put(self, key: str, data: bytes) -> None:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

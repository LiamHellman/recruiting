"""Where uploaded files live: a gitignored local folder in dev; an encrypted bucket in prod
(Phase 8.8 / 9.1 add that implementation behind the same interface)."""

import re
from pathlib import Path
from typing import Protocol

_KEY = re.compile(r"^[A-Za-z0-9-]+(?:/[A-Za-z0-9-]+)*$")


class FileStore(Protocol):
    def put(self, key: str, data: bytes) -> None: ...

    def get(self, key: str) -> bytes: ...

    def delete(self, key: str) -> None: ...


class LocalFileStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def _path(self, key: str) -> Path:
        # Keys are built from UUIDs; anything else (e.g. "..") is a bug.
        if not _KEY.match(key):
            raise ValueError(f"bad storage key {key!r}")
        return self._root / key

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(0o600)

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


class InMemoryFileStore:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def put(self, key: str, data: bytes) -> None:
        self.files[key] = data

    def get(self, key: str) -> bytes:
        return self.files[key]

    def delete(self, key: str) -> None:
        self.files.pop(key, None)

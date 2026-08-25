"""Content-addressed cache for expensive/regenerable outputs.

Used by research, image, TTS, chart, and map generation so that a retried
or resumed job does not re-pay for work it already completed successfully.
Keys are derived from a stable hash of the generation parameters.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Optional


def make_cache_key(namespace: str, params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:24]
    return f"{namespace}:{digest}"


class CacheStore:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.cache_dir / "index.json"
        self._index: dict[str, str] = {}
        self._load_index()

    def _load_index(self) -> None:
        if self._index_path.exists():
            try:
                self._index = json.loads(self._index_path.read_text())
            except Exception:
                self._index = {}

    def _save_index(self) -> None:
        self._index_path.write_text(json.dumps(self._index, indent=2))

    def get(self, key: str) -> Optional[str]:
        path = self._index.get(key)
        if path and Path(path).exists():
            return path
        return None

    def put_file(self, key: str, source_path: str, dest_filename: Optional[str] = None) -> str:
        dest_filename = dest_filename or Path(source_path).name
        dest_path = self.cache_dir / f"{key.replace(':', '_')}_{dest_filename}"
        if Path(source_path).resolve() != dest_path.resolve():
            shutil.copy2(source_path, dest_path)
        self._index[key] = str(dest_path)
        self._save_index()
        return str(dest_path)

    def put_json(self, key: str, data: dict) -> str:
        dest_path = self.cache_dir / f"{key.replace(':', '_')}.json"
        dest_path.write_text(json.dumps(data, indent=2, default=str))
        self._index[key] = str(dest_path)
        self._save_index()
        return str(dest_path)

    def get_json(self, key: str) -> Optional[dict]:
        path = self.get(key)
        if path is None:
            return None
        try:
            return json.loads(Path(path).read_text())
        except Exception:
            return None

    def clear(self) -> None:
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index = {}

"""On-disk project storage and checkpointing.

Layout (see spec section 9):

projects/<project_id>/{research,script,scenes,assets,audio,captions,
                       renders,shorts,thumbnails,metadata,qa,licenses,logs}/

Checkpoints are plain JSON files written atomically (write to a temp file,
then os.replace) so a crash mid-write can never corrupt a checkpoint that
was already good. `ProjectStore.load_checkpoint` returning None means
"this stage has not completed yet" — callers use that to resume.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional

from core.exceptions import CheckpointError
from core.logging import get_logger

logger = get_logger(__name__)

SUBDIRS = [
    "research", "script", "scenes", "assets", "audio", "captions",
    "renders", "shorts", "thumbnails", "metadata", "qa", "licenses", "logs",
    "cache", "tmp",
]

CHECKPOINT_FILES = [
    "research.json", "facts.json", "story.json", "script.json", "scenes.json",
    "assets.json", "voice.json", "music.json", "render_manifest.json", "qa.json",
    "state.json", "config.json",
]


class ProjectStore:
    def __init__(self, root: str, project_id: str):
        self.root = Path(root)
        self.project_id = project_id
        self.project_dir = self.root / project_id
        self._ensure_layout()

    def _ensure_layout(self) -> None:
        self.project_dir.mkdir(parents=True, exist_ok=True)
        for sub in SUBDIRS:
            (self.project_dir / sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Generic checkpoint read/write
    # ------------------------------------------------------------------

    def _checkpoint_path(self, name: str) -> Path:
        if not name.endswith(".json"):
            name = f"{name}.json"
        return self.project_dir / name

    def save_checkpoint(self, name: str, data: dict[str, Any]) -> str:
        path = self._checkpoint_path(name)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp_path, path)
        except Exception as exc:
            raise CheckpointError(f"Failed to save checkpoint {name}: {exc}") from exc
        logger.info("Saved checkpoint %s -> %s", name, path)
        return str(path)

    def load_checkpoint(self, name: str) -> Optional[dict[str, Any]]:
        path = self._checkpoint_path(name)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            raise CheckpointError(f"Failed to load checkpoint {name}: {exc}") from exc

    def has_checkpoint(self, name: str) -> bool:
        return self._checkpoint_path(name).exists()

    def delete_checkpoint(self, name: str) -> None:
        path = self._checkpoint_path(name)
        if path.exists():
            path.unlink()

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------

    def dir_for(self, sub: str) -> Path:
        assert sub in SUBDIRS, f"Unknown project subdir: {sub}"
        return self.project_dir / sub

    def path_in(self, sub: str, filename: str) -> str:
        return str(self.dir_for(sub) / filename)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clean_temp(self) -> None:
        tmp_dir = self.dir_for("tmp")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

    def completed_stages(self) -> list[str]:
        return [name for name in CHECKPOINT_FILES if self.has_checkpoint(name)]

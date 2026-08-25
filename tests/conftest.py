"""Shared pytest fixtures for the UDS test suite."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture()
def tmp_projects_root(tmp_path):
    root = tmp_path / "projects"
    root.mkdir(parents=True, exist_ok=True)
    yield str(root)
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture()
def sample_topic() -> str:
    return "How a Small Technology Company Became a Global Leader"

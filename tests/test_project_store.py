from __future__ import annotations

from storage.project_store import ProjectStore


def test_layout_created(tmp_projects_root):
    store = ProjectStore(root=tmp_projects_root, project_id="proj_test")
    for sub in ["research", "script", "scenes", "renders", "shorts"]:
        assert store.dir_for(sub).exists()


def test_save_and_load_checkpoint_round_trip(tmp_projects_root):
    store = ProjectStore(root=tmp_projects_root, project_id="proj_test")
    data = {"topic": "Test Topic", "value": 42}
    store.save_checkpoint("research.json", data)
    loaded = store.load_checkpoint("research.json")
    assert loaded == data


def test_missing_checkpoint_returns_none(tmp_projects_root):
    store = ProjectStore(root=tmp_projects_root, project_id="proj_test")
    assert store.load_checkpoint("nonexistent.json") is None
    assert store.has_checkpoint("nonexistent.json") is False


def test_completed_stages_reflects_saved_checkpoints(tmp_projects_root):
    store = ProjectStore(root=tmp_projects_root, project_id="proj_test")
    store.save_checkpoint("research.json", {"a": 1})
    store.save_checkpoint("script.json", {"b": 2})
    completed = store.completed_stages()
    assert "research.json" in completed
    assert "script.json" in completed
    assert "scenes.json" not in completed


def test_delete_checkpoint(tmp_projects_root):
    store = ProjectStore(root=tmp_projects_root, project_id="proj_test")
    store.save_checkpoint("research.json", {"a": 1})
    store.delete_checkpoint("research.json")
    assert store.has_checkpoint("research.json") is False


def test_atomic_write_survives_partial_failure_simulation(tmp_projects_root):
    store = ProjectStore(root=tmp_projects_root, project_id="proj_test")
    store.save_checkpoint("research.json", {"a": 1})
    # Simulate a crash mid-write: a stray .tmp file should never be picked up.
    tmp_path = store._checkpoint_path("research.json").with_suffix(".json.tmp")
    tmp_path.write_text("{corrupt")
    loaded = store.load_checkpoint("research.json")
    assert loaded == {"a": 1}

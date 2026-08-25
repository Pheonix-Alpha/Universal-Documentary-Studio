from __future__ import annotations

from storage.cache_store import CacheStore, make_cache_key


def test_make_cache_key_deterministic():
    key1 = make_cache_key("image", {"prompt": "a cat", "size": 512})
    key2 = make_cache_key("image", {"size": 512, "prompt": "a cat"})
    assert key1 == key2  # order-independent


def test_make_cache_key_differs_for_different_params():
    key1 = make_cache_key("image", {"prompt": "a cat"})
    key2 = make_cache_key("image", {"prompt": "a dog"})
    assert key1 != key2


def test_put_and_get_json(tmp_path):
    cache = CacheStore(cache_dir=str(tmp_path / "cache"))
    key = make_cache_key("research", {"topic": "x"})
    cache.put_json(key, {"result": 123})
    loaded = cache.get_json(key)
    assert loaded == {"result": 123}


def test_get_returns_none_for_missing_key(tmp_path):
    cache = CacheStore(cache_dir=str(tmp_path / "cache"))
    assert cache.get("missing:key") is None
    assert cache.get_json("missing:key") is None


def test_put_file_copies_and_indexes(tmp_path):
    src = tmp_path / "source.txt"
    src.write_text("hello")
    cache = CacheStore(cache_dir=str(tmp_path / "cache"))
    key = make_cache_key("file", {"id": 1})
    dest = cache.put_file(key, str(src))
    assert cache.get(key) == dest


def test_clear_removes_all_entries(tmp_path):
    cache = CacheStore(cache_dir=str(tmp_path / "cache"))
    key = make_cache_key("x", {"a": 1})
    cache.put_json(key, {"v": 1})
    cache.clear()
    assert cache.get_json(key) is None

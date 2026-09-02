import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# API keys
#
# Every key starts out seeded from its environment variable (handy for
# `os.environ["ANTHROPIC_API_KEY"] = "..."` in a Colab cell before
# launching), but can also be entered live from the "API Keys" tab in the
# UI. Other modules must read these via get_key()/is_key_set() at the point
# of use -- NOT `from app.config import ANTHROPIC_API_KEY` -- so a key
# entered in the UI takes effect immediately without restarting the app.
# --------------------------------------------------------------------------
KEY_SPECS = [
    {
        "id": "ANTHROPIC_API_KEY",
        "label": "Anthropic",
        "note": "Smarter scene splitting & search-query generation via Claude. "
                "Falls back to rule-based logic without it.",
        "signup_url": "https://console.anthropic.com/settings/keys",
    },
    {
        "id": "UNSPLASH_ACCESS_KEY",
        "label": "Unsplash",
        "note": "Extra image source.",
        "signup_url": "https://unsplash.com/developers",
    },
    {
        "id": "FLICKR_API_KEY",
        "label": "Flickr",
        "note": "Extra image/video source.",
        "signup_url": "https://www.flickr.com/services/apps/create/",
    },
    {
        "id": "PEXELS_API_KEY",
        "label": "Pexels",
        "note": "Extra image/video source.",
        "signup_url": "https://www.pexels.com/api/",
    },
    {
        "id": "PIXABAY_API_KEY",
        "label": "Pixabay",
        "note": "Extra image/video source.",
        "signup_url": "https://pixabay.com/api/docs/",
    },
]

_keys = {spec["id"]: os.environ.get(spec["id"], "") for spec in KEY_SPECS}


def get_key(key_id: str) -> str:
    return _keys.get(key_id, "")


def set_key(key_id: str, value: str):
    if key_id not in _keys:
        raise KeyError(f"Unknown API key id: {key_id}")
    _keys[key_id] = (value or "").strip()


def is_key_set(key_id: str) -> bool:
    return bool(_keys.get(key_id))


# Snapshot constants kept for backwards compatibility. NOTE: these only
# reflect the environment variable at import time -- code that should react
# to a key entered later in the UI must call get_key()/is_key_set() instead.
ANTHROPIC_API_KEY = _keys["ANTHROPIC_API_KEY"]
UNSPLASH_ACCESS_KEY = _keys["UNSPLASH_ACCESS_KEY"]
FLICKR_API_KEY = _keys["FLICKR_API_KEY"]
PEXELS_API_KEY = _keys["PEXELS_API_KEY"]
PIXABAY_API_KEY = _keys["PIXABAY_API_KEY"]
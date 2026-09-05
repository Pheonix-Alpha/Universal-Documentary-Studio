"""
Persistent Google Drive cache for Universal Documentary Studio.

Caches:
- Enhanced script
- Production Bible
- Individual generated video clips

Cache is keyed by the original story hash.
"""

import os
import hashlib
import pickle
import base64
from typing import Optional, Dict, Any

# ============================================================
# GOOGLE DRIVE CACHE
# ============================================================

DRIVE_ROOT = "/content/drive/MyDrive/Universal-Documentary-Studio"

CACHE_ROOT = os.path.join(DRIVE_ROOT, "cache")

SCRIPTS_DIR = os.path.join(CACHE_ROOT, "scripts")

BIBLES_DIR = os.path.join(CACHE_ROOT, "production_bibles")

SCENES_DIR = os.path.join(CACHE_ROOT, "scenes")

# ============================================================
# DRIVE AVAILABILITY
# ============================================================

def is_drive_available() -> bool:
    """
    Check whether Google Drive is actually mounted.

    /content/drive may exist even when Drive is not mounted,
    so we specifically check for MyDrive.
    """
    return os.path.isdir("/content/drive/MyDrive")


# ============================================================
# INITIALIZE
# ============================================================

def initialize_cache() -> bool:
    """
    Create the Drive cache directory structure.

    Returns False when Google Drive is unavailable.
    This is not considered a fatal application error.
    """

    if not is_drive_available():
        print("ℹ️ Google Drive is not mounted.")
        print("   Persistent Drive cache disabled.")
        return False

    try:
        os.makedirs(SCRIPTS_DIR, exist_ok=True)
        os.makedirs(BIBLES_DIR, exist_ok=True)
        os.makedirs(SCENES_DIR, exist_ok=True)

        print("✅ Drive cache initialized")
        print(f"   📁 {CACHE_ROOT}")

        return True

    except Exception as e:
        print(f"⚠️ Cache initialization failed: {e}")
        return False
# ============================================================
# STORY HASH
# ============================================================


def get_story_hash(story: str) -> str:
    """
    Generate a stable ID for the story.

    Same story -> same hash.
    """

    normalized = story.strip()

    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


# ============================================================
# SCRIPT CACHE
# ============================================================


def get_script_path(story: str) -> str:

    return os.path.join(SCRIPTS_DIR, f"{get_story_hash(story)}.txt")


def save_script(story: str, script: str) -> bool:

    try:

        path = get_script_path(story)

        with open(path, "w", encoding="utf-8") as f:
            f.write(script)

        print(f"💾 Script saved to Drive: {path}")

        return True

    except Exception as e:

        print(f"⚠️ Failed to save script: {e}")

        return False


def load_script(story: str) -> Optional[str]:

    path = get_script_path(story)

    if not os.path.exists(path):
        return None

    try:

        with open(path, "r", encoding="utf-8") as f:

            script = f.read()

        print(f"⚡ Cached script found: {path}")

        return script

    except Exception as e:

        print(f"⚠️ Failed to load cached script: {e}")

        return None


# ============================================================
# PRODUCTION BIBLE CACHE
# ============================================================


def get_bible_path(story: str) -> str:

    return os.path.join(BIBLES_DIR, f"{get_story_hash(story)}.pkl")


def save_brain_result(story: str, bible: Any, enhanced_script: str) -> bool:
    """
    Save the complete local-brain result.

    Pickle preserves the ProductionBible object exactly,
    allowing us to restore it without reconstructing
    its dataclass manually.
    """

    try:

        path = get_bible_path(story)

        data = {
            "story_hash": get_story_hash(story),
            "enhanced_script": enhanced_script,
            "bible": bible,
        }

        with open(path, "wb") as f:

            pickle.dump(data, f)

        # Also save human-readable script
        save_script(story, enhanced_script)

        print(f"💾 Production Bible saved to Drive: {path}")

        return True

    except Exception as e:

        print(f"⚠️ Failed to save brain cache: {e}")

        return False


def load_brain_result(story: str) -> Optional[Dict[str, Any]]:

    path = get_bible_path(story)

    if not os.path.exists(path):
        return None

    try:

        with open(path, "rb") as f:

            data = pickle.load(f)

        print(f"⚡ Cached Production Bible found: {path}")

        return data

    except Exception as e:

        print(f"⚠️ Failed to load brain cache: {e}")

        return None


# ============================================================
# SCENE CACHE
# ============================================================


def get_scene_directory(story: str) -> str:

    if not is_drive_available():
        raise RuntimeError("Google Drive is not mounted.")

    directory = os.path.join(SCENES_DIR, get_story_hash(story))

    os.makedirs(directory, exist_ok=True)

    return directory

def get_scene_path(story: str, scene_id: str) -> str:

    safe_scene_id = str(scene_id).replace("/", "_")

    return os.path.join(get_scene_directory(story), f"{safe_scene_id}.mp4")


def scene_exists(story: str, scene_id: str) -> bool:

    path = get_scene_path(story, scene_id)

    return os.path.exists(path) and os.path.getsize(path) > 0


# ============================================================
# SAVE VIDEO CLIP
# ============================================================


def save_scene(story: str, scene_id: str, video_data: str) -> Optional[str]:
    """
    Save completed video clip to Google Drive.

    video_data is expected to be base64 encoded.
    """

    if not video_data:
        return None

    try:

        path = get_scene_path(story, scene_id)

        # Handle data URI:
        # data:video/mp4;base64,AAAA...
        if "," in video_data:
            video_data = video_data.split(",", 1)[1]

        video_bytes = base64.b64decode(video_data)

        with open(path, "wb") as f:

            f.write(video_bytes)

        print(f"💾 Scene {scene_id} saved to Drive: {path}")

        return path

    except Exception as e:

        print(f"⚠️ Failed to save Scene {scene_id}: {e}")

        return None


# ============================================================
# LOAD VIDEO CLIP
# ============================================================


def load_scene(story: str, scene_id: str) -> Optional[str]:

    path = get_scene_path(story, scene_id)

    if not os.path.exists(path):
        return None

    try:

        with open(path, "rb") as f:

            video_bytes = f.read()

        video_data = base64.b64encode(video_bytes).decode("utf-8")

        print(f"⚡ Cached Scene {scene_id} loaded")

        return video_data

    except Exception as e:

        print(f"⚠️ Failed to load Scene {scene_id}: {e}")

        return None

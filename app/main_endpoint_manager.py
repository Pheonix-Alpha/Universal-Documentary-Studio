"""
Main endpoint discovery through Google Drive.

Main writes its current public API endpoint to Google Drive.
Workers can later read this file to discover the Main API.
"""

import json
import os
import time


ENDPOINT_FILENAME = "main_endpoint.json"
MAX_ENDPOINT_AGE = 60


def is_drive_available() -> bool:
    """Return True when Google Drive is mounted."""
    return os.path.isdir("/content/drive/MyDrive")


def get_endpoint_path() -> str:
    """Return the path used to store the Main endpoint."""
    if not is_drive_available():
        raise RuntimeError("Google Drive is not mounted.")

    return os.path.join(
        "/content/drive/MyDrive",
        ENDPOINT_FILENAME,
    )


def write_main_endpoint(
    url: str,
    runtime_id: str,
) -> bool:
    """Write or update the current Main API endpoint."""

    if not is_drive_available():
        print("ℹ️ Google Drive is not mounted.")
        print("   Main endpoint discovery through Drive disabled.")
        return False

    if not url:
        print("❌ Cannot write Main endpoint without URL.")
        return False

    data = {
        "url": url.rstrip("/"),
        "runtime_id": runtime_id,
        "updated_at": time.time(),
    }

    try:
        path = get_endpoint_path()

        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

        print(f"✅ Main endpoint written to Drive: {path}")
        print(f"   URL: {data['url']}")

        return True

    except Exception as e:
        print(f"❌ Failed to write Main endpoint: {e}")
        return False


def read_main_endpoint():
    """
    Read the latest Main API endpoint from Google Drive.

    The endpoint is NOT rejected just because it is old.
    The caller is responsible for testing whether the URL
    is actually reachable.

    Returns:
        dict | None
    """

    if not is_drive_available():
        return None

    try:
        path = get_endpoint_path()

        if not os.path.exists(path):
            print("ℹ️ Main endpoint file not found.")
            return None

        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        url = data.get("url")
        updated_at = data.get("updated_at")

        if not url or not updated_at:
            print("⚠️ Main endpoint file is invalid.")
            return None

        age = time.time() - float(updated_at)

        data["age"] = age
        data["is_stale"] = age > MAX_ENDPOINT_AGE

        if data["is_stale"]:
            print(
                f"⚠️ Main endpoint is old "
                f"({age:.1f}s old)."
            )
            print("   URL will still be tested before use.")
        else:
            print(
                f"✅ Main endpoint found "
                f"({age:.1f}s old)."
            )

        return data

    except Exception as e:
        print(f"❌ Failed to read Main endpoint: {e}")
        return None
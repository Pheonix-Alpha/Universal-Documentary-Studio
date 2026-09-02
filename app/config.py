import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Optional -- set these as environment variables before launching for
# LLM-based scene analysis / query generation and extra media sources.
# The app works without any of them (rule-based fallback + the no-key
# media sources: Wikimedia, NASA, Internet Archive, Met Museum, Openverse,
# Library of Congress, DuckDuckGo).
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
UNSPLASH_ACCESS_KEY = os.environ.get("UNSPLASH_ACCESS_KEY", "")
FLICKR_API_KEY = os.environ.get("FLICKR_API_KEY", "")
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")
PIXABAY_API_KEY = os.environ.get("PIXABAY_API_KEY", "")
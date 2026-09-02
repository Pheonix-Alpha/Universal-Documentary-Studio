"""Query string -> list of candidate {url, source, title} dicts, from multiple sources."""
import time

import requests

from app.config import UNSPLASH_ACCESS_KEY

# Wikimedia (and most APIs) throttle/block requests with no descriptive
# User-Agent -- without this you get an HTML error page back instead of
# JSON, which is what caused "Expecting value: line 1 column 1" errors.
USER_AGENT = "ResearchAIStoryboard/1.0 (educational prototype; https://github.com/)"
HEADERS = {"User-Agent": USER_AGENT}


def search_wikimedia(query: str, limit: int = 8):
    """Wikimedia Commons is a strong source for historical / documentary accuracy."""
    url = "https://commons.wikimedia.org/w/api.php"
    # Commons' search works better on a few keywords than a full sentence --
    # trim to the first ~8 words to raise the odds of getting real hits.
    short_query = " ".join(query.split()[:8])
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": short_query,
        "gsrlimit": limit,
        "gsrnamespace": 6,  # File: namespace
        "prop": "imageinfo",
        "iiprop": "url",
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            print(f"[image_sources] Wikimedia HTTP {r.status_code} for '{short_query}': {r.text[:150]}")
            return []
        try:
            data = r.json()
        except ValueError:
            print(f"[image_sources] Wikimedia returned non-JSON for '{short_query}': {r.text[:150]}")
            return []
        pages = data.get("query", {}).get("pages", {})
        results = []
        for p in pages.values():
            info = (p.get("imageinfo") or [{}])[0]
            if info.get("url"):
                results.append({"url": info["url"], "source": "wikimedia", "title": p.get("title", "")})
        return results
    except Exception as e:  # noqa: BLE001
        print(f"[image_sources] Wikimedia search failed for '{short_query}': {e}")
        return []


def search_unsplash(query: str, limit: int = 8):
    """Generic, high-quality stock photos -- good for visual variety, weaker for historical accuracy."""
    if not UNSPLASH_ACCESS_KEY:
        return []
    try:
        r = requests.get(
            "https://api.unsplash.com/search/photos",
            params={"query": query, "per_page": limit},
            headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"},
            timeout=10,
        )
        data = r.json()
        return [
            {"url": item["urls"]["regular"], "source": "unsplash", "title": item.get("alt_description") or ""}
            for item in data.get("results", [])
            if item.get("urls", {}).get("regular")
        ]
    except Exception as e:  # noqa: BLE001
        print(f"[image_sources] Unsplash search failed for '{query}': {e}")
        return []


def _get_ddgs_class():
    """The package was renamed duckduckgo_search -> ddgs. Support either."""
    try:
        from ddgs import DDGS  # new name
        return DDGS
    except ImportError:
        from duckduckgo_search import DDGS  # old name, still works but warns
        return DDGS


def search_duckduckgo(query: str, limit: int = 8, max_retries: int = 3):
    """No-API-key general web image search. Its unofficial endpoint rate-limits
    aggressively under bursts, so this retries with backoff and treats
    rate-limit errors as 'no results this time' rather than a hard failure."""
    try:
        DDGS = _get_ddgs_class()
    except Exception as e:  # noqa: BLE001
        print(f"[image_sources] duckduckgo/ddgs not available: {e}")
        return []

    for attempt in range(max_retries):
        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.images(query, max_results=limit):
                    if r.get("image"):
                        results.append({"url": r["image"], "source": "duckduckgo", "title": r.get("title") or ""})
            return results
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "atelimit" in msg or "403" in msg or "202" in msg:
                wait = 2 * (attempt + 1)
                print(f"[image_sources] DuckDuckGo rate-limited for '{query}', retrying in {wait}s...")
                time.sleep(wait)
                continue
            print(f"[image_sources] DuckDuckGo search failed for '{query}': {e}")
            return []
    print(f"[image_sources] DuckDuckGo still rate-limited after {max_retries} attempts for '{query}', skipping.")
    return []


def gather_candidates(query: str, limit_per_source: int = 8, use_duckduckgo: bool = True):
    candidates = []
    candidates += search_wikimedia(query, limit_per_source)
    candidates += search_unsplash(query, limit_per_source)
    if use_duckduckgo:
        candidates += search_duckduckgo(query, limit_per_source)

    seen, unique = set(), []
    for c in candidates:
        if c["url"] and c["url"] not in seen:
            seen.add(c["url"])
            unique.append(c)
    return unique

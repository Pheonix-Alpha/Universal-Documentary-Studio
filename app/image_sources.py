"""Query string -> list of candidate {url, source, title} dicts, from multiple sources."""
import requests

from app.config import UNSPLASH_ACCESS_KEY


def search_wikimedia(query: str, limit: int = 8):
    """Wikimedia Commons is a strong source for historical / documentary accuracy."""
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": limit,
        "gsrnamespace": 6,  # File: namespace
        "prop": "imageinfo",
        "iiprop": "url",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        results = []
        for p in pages.values():
            info = (p.get("imageinfo") or [{}])[0]
            if info.get("url"):
                results.append({"url": info["url"], "source": "wikimedia", "title": p.get("title", "")})
        return results
    except Exception as e:  # noqa: BLE001
        print(f"[image_sources] Wikimedia search failed for '{query}': {e}")
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


def search_duckduckgo(query: str, limit: int = 8):
    """No-API-key general web image search, used as a broad fallback source."""
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.images(query, max_results=limit):
                if r.get("image"):
                    results.append({"url": r["image"], "source": "duckduckgo", "title": r.get("title") or ""})
        return results
    except Exception as e:  # noqa: BLE001
        print(f"[image_sources] DuckDuckGo search failed for '{query}': {e}")
        return []


def gather_candidates(query: str, limit_per_source: int = 8):
    candidates = []
    candidates += search_wikimedia(query, limit_per_source)
    candidates += search_unsplash(query, limit_per_source)
    candidates += search_duckduckgo(query, limit_per_source)

    seen, unique = set(), []
    for c in candidates:
        if c["url"] and c["url"] not in seen:
            seen.add(c["url"])
            unique.append(c)
    return unique

"""
Free media search across multiple sources, returning a unified schema:

    {
      "url": <link to the full-resolution image, or a page/URL for the video>,
      "thumbnail_url": <an image URL usable for both previews and CLIP scoring>,
      "source": "wikimedia" | "nasa" | "internet_archive" | "met_museum" |
                 "openverse" | "loc" | "flickr" | "pexels" | "pixabay" | "duckduckgo",
      "title": str,
      "media_type": "image" | "video",
    }

No-key sources (work out of the box): Wikimedia Commons, NASA Image & Video
Library, Internet Archive, The Met Open Access, Openverse, Library of
Congress.

Optional-key sources (set the matching env var to enable -- all have free
tiers, just need a free account): Flickr (FLICKR_API_KEY), Pexels
(PEXELS_API_KEY), Pixabay (PIXABAY_API_KEY).

DuckDuckGo image search is kept as a broad no-key fallback, but its
unofficial endpoint rate-limits hard, so it's used sparingly by the caller
(see MAX_DDG_QUERIES_PER_SCENE in pipeline.py).

Other free sources worth adding later: Smithsonian Open Access and
Europeana both have generous free APIs but require registering for a key;
Rijksmuseum and the NYPL Digital Collections are similar. None are wired up
here to keep the no-key path simple, but they'd slot in the same way as
Flickr/Pexels/Pixabay below.
"""
import time

import requests

from app.config import FLICKR_API_KEY, PEXELS_API_KEY, PIXABAY_API_KEY

USER_AGENT = "ResearchAIStoryboard/1.0 (educational prototype; https://github.com/)"
HEADERS = {"User-Agent": USER_AGENT}


def _candidate(url, thumbnail_url, source, title, media_type="image"):
    return {
        "url": url,
        "thumbnail_url": thumbnail_url or url,
        "source": source,
        "title": title or "",
        "media_type": media_type,
    }


# ---------------------------------------------------------------- Wikimedia
def search_wikimedia(query: str, limit: int = 6, media_types=("image", "video")):
    """Wikimedia Commons -- free, no key. Strong for historical/documentary
    accuracy, and its File: namespace holds both photos and short clips."""
    short_query = " ".join(query.split()[:8])
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": short_query,
        "gsrlimit": limit,
        "gsrnamespace": 6,  # File:
        "prop": "imageinfo",
        "iiprop": "url|mime",
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            print(f"[media_sources] Wikimedia HTTP {r.status_code} for '{short_query}': {r.text[:150]}")
            return []
        try:
            data = r.json()
        except ValueError:
            print(f"[media_sources] Wikimedia returned non-JSON for '{short_query}': {r.text[:150]}")
            return []
        pages = data.get("query", {}).get("pages", {})
        results = []
        for p in pages.values():
            info = (p.get("imageinfo") or [{}])[0]
            file_url = info.get("url")
            if not file_url:
                continue
            mtype = "video" if info.get("mime", "").startswith("video") else "image"
            if mtype not in media_types:
                continue
            results.append(_candidate(file_url, file_url, "wikimedia", p.get("title", ""), mtype))
        return results
    except Exception as e:  # noqa: BLE001
        print(f"[media_sources] Wikimedia search failed for '{short_query}': {e}")
        return []


# --------------------------------------------------------------------- NASA
def search_nasa(query: str, limit: int = 6, media_types=("image", "video")):
    """NASA Image and Video Library -- free, no API key needed."""
    type_param = ",".join(t for t in media_types if t in ("image", "video"))
    if not type_param:
        return []
    try:
        r = requests.get(
            "https://images-api.nasa.gov/search",
            params={"q": query, "media_type": type_param},
            headers=HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        items = r.json().get("collection", {}).get("items", [])[:limit]
        results = []
        for item in items:
            data0 = (item.get("data") or [{}])[0]
            links = item.get("links") or []
            thumb = next((l["href"] for l in links if l.get("rel") == "preview"), None)
            mtype = data0.get("media_type", "image")
            if mtype not in media_types or not thumb:
                continue
            nasa_id = data0.get("nasa_id", "")
            page_url = f"https://images.nasa.gov/details/{nasa_id}" if nasa_id else thumb
            results.append(_candidate(page_url, thumb, "nasa", data0.get("title", ""), mtype))
        return results
    except Exception as e:  # noqa: BLE001
        print(f"[media_sources] NASA search failed for '{query}': {e}")
        return []


# ---------------------------------------------------------- Internet Archive
def search_internet_archive(query: str, limit: int = 6, media_types=("image", "video")):
    """archive.org -- free, no key. Huge public-domain/CC archive of photos and film."""
    ia_type_map = {"image": "image", "video": "movies"}
    wanted = [ia_type_map[t] for t in media_types if t in ia_type_map]
    if not wanted:
        return []
    try:
        mediatype_query = " OR ".join(f"mediatype:{t}" for t in wanted)
        r = requests.get(
            "https://archive.org/advancedsearch.php",
            params={
                "q": f"({query}) AND ({mediatype_query})",
                "fl[]": ["identifier", "title", "mediatype"],
                "rows": limit,
                "output": "json",
            },
            headers=HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        docs = r.json().get("response", {}).get("docs", [])
        results = []
        for d in docs:
            ident = d.get("identifier")
            if not ident:
                continue
            mtype = "video" if d.get("mediatype") == "movies" else "image"
            thumb = f"https://archive.org/services/img/{ident}"
            page_url = f"https://archive.org/details/{ident}"
            results.append(_candidate(page_url, thumb, "internet_archive", d.get("title", ""), mtype))
        return results
    except Exception as e:  # noqa: BLE001
        print(f"[media_sources] Internet Archive search failed for '{query}': {e}")
        return []


# --------------------------------------------------------------- Met Museum
def search_met_museum(query: str, limit: int = 6, media_types=("image", "video")):
    """The Met's Open Access API -- free, no key. Public-domain artwork images only (no video)."""
    if "image" not in media_types:
        return []
    try:
        r = requests.get(
            "https://collectionapi.metmuseum.org/public/collection/v1/search",
            params={"q": query, "hasImages": "true"},
            headers=HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        ids = (r.json().get("objectIDs") or [])[:limit]
        results = []
        for obj_id in ids:
            try:
                r2 = requests.get(
                    f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{obj_id}",
                    headers=HEADERS,
                    timeout=10,
                )
                obj = r2.json()
                img = obj.get("primaryImageSmall") or obj.get("primaryImage")
                if img:
                    results.append(
                        _candidate(obj.get("primaryImage") or img, img, "met_museum", obj.get("title", ""), "image")
                    )
            except Exception:  # noqa: BLE001
                continue
        return results
    except Exception as e:  # noqa: BLE001
        print(f"[media_sources] Met Museum search failed for '{query}': {e}")
        return []


# --------------------------------------------------------------- Openverse
def search_openverse(query: str, limit: int = 6, media_types=("image", "video")):
    """Openverse aggregates hundreds of millions of CC-licensed images (Flickr,
    Wikimedia, museums, etc). Free, no key. Images only for now."""
    if "image" not in media_types:
        return []
    try:
        r = requests.get(
            "https://api.openverse.org/v1/images/",
            params={"q": query, "page_size": limit},
            headers=HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        results = []
        for item in r.json().get("results", []):
            url = item.get("url")
            if url:
                results.append(_candidate(url, item.get("thumbnail") or url, "openverse", item.get("title", ""), "image"))
        return results
    except Exception as e:  # noqa: BLE001
        print(f"[media_sources] Openverse search failed for '{query}': {e}")
        return []


# --------------------------------------------------------- Library of Congress
def search_loc(query: str, limit: int = 6, media_types=("image", "video")):
    """loc.gov -- free, no key. Huge photo/print/film archive."""
    try:
        r = requests.get(
            "https://www.loc.gov/search/",
            params={"q": query, "fo": "json", "c": limit},
            headers=HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        results = []
        for item in r.json().get("results", []):
            img = item.get("image_url")
            thumb = img[-1] if isinstance(img, list) and img else (img if isinstance(img, str) else None)
            if not thumb:
                continue
            formats = " ".join(item.get("original_format", []) or []).lower()
            mtype = "video" if "moving" in formats or "film" in formats else "image"
            if mtype not in media_types:
                continue
            page_url = item.get("id") or thumb
            results.append(_candidate(page_url, thumb, "loc", item.get("title", ""), mtype))
        return results[:limit]
    except Exception as e:  # noqa: BLE001
        print(f"[media_sources] Library of Congress search failed for '{query}': {e}")
        return []


# ------------------------------------------------------------------- Flickr
def search_flickr(query: str, limit: int = 6, media_types=("image", "video")):
    """Optional -- set FLICKR_API_KEY (free): https://www.flickr.com/services/apps/create/"""
    if not FLICKR_API_KEY:
        return []
    media_param = "all" if len(media_types) > 1 else ("photos" if "image" in media_types else "videos")
    try:
        r = requests.get(
            "https://api.flickr.com/services/rest/",
            params={
                "method": "flickr.photos.search",
                "api_key": FLICKR_API_KEY,
                "text": query,
                "media": media_param,
                "license": "1,2,3,4,5,6,7,8,9,10",  # Creative Commons / public domain / no known restriction
                "per_page": limit,
                "format": "json",
                "nojsoncallback": 1,
                "extras": "url_c,media",
            },
            headers=HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        photos = r.json().get("photos", {}).get("photo", [])
        results = []
        for p in photos:
            thumb = p.get("url_c")
            if not thumb:
                continue
            mtype = "video" if p.get("media") == "video" else "image"
            page_url = f"https://www.flickr.com/photos/{p.get('owner')}/{p.get('id')}"
            results.append(_candidate(page_url, thumb, "flickr", p.get("title", ""), mtype))
        return results
    except Exception as e:  # noqa: BLE001
        print(f"[media_sources] Flickr search failed for '{query}': {e}")
        return []


# ------------------------------------------------------------------ Pexels
def search_pexels(query: str, limit: int = 6, media_types=("image", "video")):
    """Optional -- set PEXELS_API_KEY (free): https://www.pexels.com/api/"""
    if not PEXELS_API_KEY:
        return []
    headers = dict(HEADERS)
    headers["Authorization"] = PEXELS_API_KEY
    results = []
    try:
        if "image" in media_types:
            r = requests.get(
                "https://api.pexels.com/v1/search",
                params={"query": query, "per_page": limit},
                headers=headers,
                timeout=10,
            )
            r.raise_for_status()
            for p in r.json().get("photos", []):
                thumb = p.get("src", {}).get("medium")
                if thumb:
                    results.append(
                        _candidate(p.get("url") or p.get("src", {}).get("original", thumb), thumb, "pexels", p.get("alt", ""), "image")
                    )
        if "video" in media_types:
            r = requests.get(
                "https://api.pexels.com/videos/search",
                params={"query": query, "per_page": limit},
                headers=headers,
                timeout=10,
            )
            r.raise_for_status()
            for v in r.json().get("videos", []):
                thumb = v.get("image")
                if thumb:
                    results.append(_candidate(v.get("url"), thumb, "pexels", "", "video"))
        return results
    except Exception as e:  # noqa: BLE001
        print(f"[media_sources] Pexels search failed for '{query}': {e}")
        return results


# ----------------------------------------------------------------- Pixabay
def search_pixabay(query: str, limit: int = 6, media_types=("image", "video")):
    """Optional -- set PIXABAY_API_KEY (free): https://pixabay.com/api/docs/"""
    if not PIXABAY_API_KEY:
        return []
    results = []
    try:
        if "image" in media_types:
            r = requests.get(
                "https://pixabay.com/api/",
                params={"key": PIXABAY_API_KEY, "q": query, "per_page": max(3, limit)},
                headers=HEADERS,
                timeout=10,
            )
            r.raise_for_status()
            for h in r.json().get("hits", []):
                thumb = h.get("webformatURL")
                if thumb:
                    results.append(
                        _candidate(h.get("largeImageURL") or thumb, thumb, "pixabay", h.get("tags", ""), "image")
                    )
        if "video" in media_types:
            r = requests.get(
                "https://pixabay.com/api/videos/",
                params={"key": PIXABAY_API_KEY, "q": query, "per_page": max(3, limit)},
                headers=HEADERS,
                timeout=10,
            )
            r.raise_for_status()
            for h in r.json().get("hits", []):
                thumb = h.get("videos", {}).get("tiny", {}).get("thumbnail") or h.get("userImageURL")
                video_url = h.get("videos", {}).get("medium", {}).get("url")
                if video_url:
                    results.append(_candidate(video_url, thumb or video_url, "pixabay", h.get("tags", ""), "video"))
        return results
    except Exception as e:  # noqa: BLE001
        print(f"[media_sources] Pixabay search failed for '{query}': {e}")
        return results


# --------------------------------------------------------------- DuckDuckGo
def _get_ddgs_class():
    """The package was renamed duckduckgo_search -> ddgs. Support either."""
    try:
        from ddgs import DDGS  # new name
        return DDGS
    except ImportError:
        from duckduckgo_search import DDGS  # old name, still works but warns
        return DDGS


def search_duckduckgo(query: str, limit: int = 6, max_retries: int = 3, media_types=("image",)):
    """No-key general web image search, kept as a broad fallback. Its
    unofficial endpoint rate-limits hard under bursts, so this retries with
    backoff and treats rate-limit errors as 'no results this time' rather
    than a hard failure. Images only -- there's no reliable no-key video
    equivalent."""
    if "image" not in media_types:
        return []
    try:
        DDGS = _get_ddgs_class()
    except Exception as e:  # noqa: BLE001
        print(f"[media_sources] duckduckgo/ddgs not available: {e}")
        return []

    for attempt in range(max_retries):
        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.images(query, max_results=limit):
                    if r.get("image"):
                        results.append(_candidate(r["image"], r["image"], "duckduckgo", r.get("title") or "", "image"))
            return results
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "atelimit" in msg or "403" in msg or "202" in msg:
                wait = 2 * (attempt + 1)
                print(f"[media_sources] DuckDuckGo rate-limited for '{query}', retrying in {wait}s...")
                time.sleep(wait)
                continue
            print(f"[media_sources] DuckDuckGo search failed for '{query}': {e}")
            return []
    print(f"[media_sources] DuckDuckGo still rate-limited after {max_retries} attempts for '{query}', skipping.")
    return []


# -------------------------------------------------------------- Aggregator
NO_KEY_SOURCES = [
    search_wikimedia,
    search_nasa,
    search_internet_archive,
    search_met_museum,
    search_openverse,
    search_loc,
]
OPTIONAL_KEY_SOURCES = [search_flickr, search_pexels, search_pixabay]


def gather_candidates(query: str, limit_per_source: int = 6, media_types=("image", "video"), use_duckduckgo: bool = True):
    candidates = []
    for fn in NO_KEY_SOURCES + OPTIONAL_KEY_SOURCES:
        candidates += fn(query, limit_per_source, media_types=media_types)
    if use_duckduckgo:
        candidates += search_duckduckgo(query, limit_per_source, media_types=media_types)

    seen, unique = set(), []
    for c in candidates:
        if c["url"] and c["url"] not in seen:
            seen.add(c["url"])
            unique.append(c)
    return unique
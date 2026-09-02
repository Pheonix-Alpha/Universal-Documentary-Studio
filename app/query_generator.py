"""Scene dict -> list of 3-4 image-search query strings."""
import json

from app.config import ANTHROPIC_API_KEY
from app.scene_analyzer import _strip_code_fence


def _fallback_queries(scene: dict):
    desc = scene.get("description", "").strip()
    people = " ".join(scene.get("people") or [])
    extra = " ".join(
        p for p in [people, scene.get("location", ""), str(scene.get("year", "")), scene.get("event", "")] if p
    ).strip()

    candidates = [desc]
    if extra:
        candidates.append(extra)
        candidates.append(f"{desc} {extra}".strip())
        if people:
            candidates.append(f"{people} {scene.get('event','')} {scene.get('year','')}".strip())

    seen, queries = set(), []
    for c in candidates:
        c = " ".join(c.split())  # collapse whitespace
        if c and c not in seen:
            seen.add(c)
            queries.append(c)
    return queries[:4]


def _generate_with_claude(scene: dict):
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = (
        "Given this scene JSON, write 3-4 short, specific image-search queries "
        "that would find historically/factually accurate photos for it "
        "(include names, places, and dates where available). "
        "Return ONLY a JSON list of strings, nothing else.\n\n"
        f"Scene:\n{json.dumps(scene)}"
    )
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    return json.loads(_strip_code_fence(text))


def generate_queries(scene: dict):
    if ANTHROPIC_API_KEY:
        try:
            qs = _generate_with_claude(scene)
            if qs:
                return qs
        except Exception as e:  # noqa: BLE001
            print(f"[query_generator] Claude query generation failed, falling back: {e}")
    return _fallback_queries(scene)

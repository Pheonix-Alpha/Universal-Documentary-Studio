"""Scene dict -> list of 3-4 image-search query strings."""
import json

from app import model_manager
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


def _generate_with_local_brain(scene: dict):
    """Uses the local LLM (see model_manager.generate_text) instead of the
    Anthropic API."""
    prompt = (
        "Given this scene JSON, write 3-4 short, specific image-search queries "
        "that would find historically/factually accurate photos for it "
        "(include names, places, and dates where available). "
        "Return ONLY a JSON list of strings, nothing else.\n\n"
        f"Scene:\n{json.dumps(scene)}"
    )
    text = model_manager.generate_text(
        user_prompt=prompt,
        system_prompt="You write concise, specific image-search queries. Output only a JSON list of strings.",
        max_new_tokens=300,
        temperature=0.3,
    )
    return json.loads(_strip_code_fence(text))


def generate_queries(scene: dict):
    try:
        qs = _generate_with_local_brain(scene)
        if qs:
            return qs
    except Exception as e:  # noqa: BLE001
        print(f"[query_generator] Local brain query generation failed, falling back: {e}")
    return _fallback_queries(scene)
"""Story -> list of scene dicts: {scene_id, description, people, location, year, event}."""
import json
import re

from app import config


def _fallback_scene_split(story: str):
    sentences = re.split(r"(?<=[.!?])\s+", story.strip())
    scenes = []
    sid = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        sid += 1
        scenes.append(
            {
                "scene_id": sid,
                "description": s,
                "people": [],
                "location": "",
                "year": "",
                "event": "",
            }
        )
    return scenes or [{"scene_id": 1, "description": story.strip(), "people": [], "location": "", "year": "", "event": ""}]


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def _analyze_with_claude(story: str):
    import anthropic

    client = anthropic.Anthropic(api_key=config.get_key("ANTHROPIC_API_KEY"))
    prompt = (
        "Break the following story into scenes for an image storyboard. "
        "For each scene extract any people, location, year and event mentioned "
        "or clearly implied. Return ONLY valid JSON: a list of objects with keys "
        "scene_id (int), description (string), people (list of strings), "
        "location (string, may be empty), year (string, may be empty), "
        "event (string, may be empty). No prose, no markdown fences.\n\n"
        f"Story:\n{story}"
    )
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    return json.loads(_strip_code_fence(text))


def analyze_story(story: str):
    if config.is_key_set("ANTHROPIC_API_KEY"):
        try:
            return _analyze_with_claude(story)
        except Exception as e:  # noqa: BLE001
            print(f"[scene_analyzer] Claude analysis failed, falling back to rule-based split: {e}")
    return _fallback_scene_split(story)
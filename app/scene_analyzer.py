"""Story -> list of scene dicts: {scene_id, description, people, location, year, event}."""
import json
import re

from app import model_manager


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


def _analyze_with_local_brain(story: str, progress_callback=None):
    """Uses the local LLM running on the main Colab's own GPU (see
    model_manager.generate_text) instead of the Anthropic API -- no key,
    no external call."""
    prompt = (
        "Break the following story into scenes for an image storyboard. "
        "For each scene extract any people, location, year and event mentioned "
        "or clearly implied. Return ONLY valid JSON: a list of objects with keys "
        "scene_id (int), description (string), people (list of strings), "
        "location (string, may be empty), year (string, may be empty), "
        "event (string, may be empty). No prose, no markdown fences.\n\n"
        f"Story:\n{story}"
    )
    text = model_manager.generate_text(
        user_prompt=prompt,
        system_prompt="You are a scene-breakdown assistant. Output only valid JSON.",
        max_new_tokens=2000,
        temperature=0.2,
        progress_callback=progress_callback,
    )
    return json.loads(_strip_code_fence(text))


def analyze_story(story: str, progress_callback=None):

    try:
        return _analyze_with_local_brain(
            story,
            progress_callback=progress_callback
        )

    except Exception as e:
        print(
            f"[scene_analyzer] Local brain analysis failed, "
            f"falling back to rule-based split: {e}"
        )

    return _fallback_scene_split(story)
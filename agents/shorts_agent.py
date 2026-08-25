"""ShortsAgent: script.json + scenes.json -> short scene plans (spec section 40).

Shorts are NOT simple crops of the long video. Each short gets its own
independent hook + payoff, built from a subset of the strongest claims,
and a scene plan re-targeted for vertical (9:16) viewing.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.logging import get_logger
from core.models import CameraMovement, Claim, Research, Scene, ScenePlan, Script, VisualType
from adapters.tts.mock_engine import estimate_duration_seconds

logger = get_logger(__name__)


@dataclass
class ShortSpec:
    short_id: str
    title: str
    scene_plan: ScenePlan


_HOOKS = [
    "Here's the part of {topic} most people never hear about.",
    "This single moment changed everything for {topic}.",
    "{topic} almost went a completely different way.",
    "Nobody expected this twist in the story of {topic}.",
    "nobody talks about how {topic} really started.",
]


class ShortsAgent:
    def run(self, research: Research, script: Script, short_count: int = 4) -> list[ShortSpec]:
        short_count = max(3, min(5, short_count))
        strongest_claims = sorted(research.claims, key=lambda c: c.confidence, reverse=True)

        shorts: list[ShortSpec] = []
        for i in range(short_count):
            claim = strongest_claims[i % len(strongest_claims)] if strongest_claims else None
            title = f"{research.topic} — Part {i + 1}"
            plan = self._build_short_scene_plan(research.topic, claim, i)
            shorts.append(ShortSpec(short_id=f"short_{i+1}", title=title, scene_plan=plan))

        logger.info("Generated %d shorts for topic=%r", len(shorts), research.topic)
        return shorts

    def _build_short_scene_plan(self, topic: str, claim: Claim | None, variant: int) -> ScenePlan:
        hook_text = _HOOKS[variant % len(_HOOKS)].format(topic=topic)
        context_text = claim.text if claim else f"A key detail about {topic}."
        payoff_text = f"That single fact reframes how we understand {topic}."

        scenes = []
        for idx, (text, visual, camera) in enumerate([
            (hook_text, VisualType.TEXT_ANIMATION, CameraMovement.SLOW_PUSH_IN),
            (context_text, VisualType.IMAGE_ANIMATION, CameraMovement.ZOOM),
            (payoff_text, VisualType.IMAGE_ANIMATION, CameraMovement.SLOW_PULL_OUT),
        ]):
            scenes.append(Scene(
                index=idx, narration=text,
                duration_seconds=round(estimate_duration_seconds(text, speed=1.05), 2),
                visual_objective=text[:60], visual_type=visual, camera_movement=camera,
                composition="centered", lighting="dramatic", transition="cut" if idx == 0 else "dissolve",
                music_mood="cinematic", captions=True,
            ))
        plan = ScenePlan(scenes=scenes)
        plan.recompute()
        return plan

"""SceneAgent: script.json -> scenes.json (spec section 20/21/22).

This is the central component: it converts narration sections into
individual scenes, each with narration, timing, and a full visual/camera
treatment. Visual type selection is scene/story driven (spec 21), not a
single forced technique, and camera movement is chosen to be motivated
rather than random (spec 22).
"""
from __future__ import annotations

import re

from adapters.tts.mock_engine import estimate_duration_seconds
from core.logging import get_logger
from core.models import (
    AssetRequirement, CameraMovement, LensProfile, Scene, ScenePlan, Script, Story,
    StoryStructure, VisualType,
)

logger = get_logger(__name__)

# Section type -> preferred visual technique(s), cycled through so
# consecutive scenes of the same section type aren't visually identical.
_SECTION_VISUALS: dict[str, list[VisualType]] = {
    "hook": [VisualType.IMAGE_ANIMATION, VisualType.TEXT_ANIMATION],
    "context": [VisualType.IMAGE_ANIMATION, VisualType.MAP, VisualType.TIMELINE],
    "setup": [VisualType.IMAGE_ANIMATION, VisualType.DOCUMENT_ANIMATION],
    "conflict": [VisualType.IMAGE_ANIMATION, VisualType.GENERATED_IMAGE],
    "investigation": [VisualType.DOCUMENT_ANIMATION, VisualType.IMAGE_ANIMATION],
    "development": [VisualType.IMAGE_ANIMATION, VisualType.CHART],
    "turning_point": [VisualType.GENERATED_IMAGE, VisualType.IMAGE_ANIMATION],
    "consequences": [VisualType.CHART, VisualType.IMAGE_ANIMATION],
    "analysis": [VisualType.DIAGRAM, VisualType.CHART],
    "conclusion": [VisualType.IMAGE_ANIMATION, VisualType.TEXT_ANIMATION],
}

_SECTION_CAMERA: dict[str, list[CameraMovement]] = {
    "hook": [CameraMovement.SLOW_PUSH_IN, CameraMovement.STATIC],
    "context": [CameraMovement.PAN_RIGHT, CameraMovement.STATIC],
    "setup": [CameraMovement.SLOW_PUSH_IN, CameraMovement.TRACKING],
    "conflict": [CameraMovement.HANDHELD, CameraMovement.ZOOM],
    "investigation": [CameraMovement.SLOW_PUSH_IN, CameraMovement.STATIC],
    "development": [CameraMovement.DOLLY, CameraMovement.PAN_LEFT],
    "turning_point": [CameraMovement.ZOOM, CameraMovement.CRANE],
    "consequences": [CameraMovement.SLOW_PULL_OUT, CameraMovement.STATIC],
    "analysis": [CameraMovement.STATIC, CameraMovement.SLOW_PUSH_IN],
    "conclusion": [CameraMovement.SLOW_PULL_OUT, CameraMovement.STATIC],
}

_MOOD_BY_SECTION = {
    "hook": "cinematic", "context": "neutral", "setup": "neutral",
    "conflict": "dramatic", "investigation": "mystery", "development": "cinematic",
    "turning_point": "dramatic", "consequences": "emotional", "analysis": "corporate",
    "conclusion": "emotional",
}

_MAX_SENTENCES_PER_SCENE = 2


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


class SceneAgent:
    def run(self, script: Script, story: Story) -> ScenePlan:
        scenes: list[Scene] = []
        index = 0

        for section in script.sections:
            sentences = _split_sentences(section.narration) or [section.narration]
            for i in range(0, len(sentences), _MAX_SENTENCES_PER_SCENE):
                chunk = " ".join(sentences[i:i + _MAX_SENTENCES_PER_SCENE])
                visuals = _SECTION_VISUALS.get(section.section_type, [VisualType.IMAGE_ANIMATION])
                cameras = _SECTION_CAMERA.get(section.section_type, [CameraMovement.STATIC])
                visual_type = visuals[index % len(visuals)]
                camera = cameras[index % len(cameras)]
                lens = LensProfile.MM50 if visual_type == VisualType.GENERATED_IMAGE else LensProfile.MM35

                scene = Scene(
                    index=index,
                    narration=chunk,
                    duration_seconds=round(estimate_duration_seconds(chunk), 2),
                    claim_ids=list(section.claim_ids),
                    visual_objective=f"Support the '{section.section_type}' beat of the story.",
                    visual_type=visual_type,
                    camera_movement=camera,
                    lens=lens,
                    composition="rule_of_thirds" if visual_type == VisualType.GENERATED_IMAGE else "centered",
                    lighting="dramatic" if section.section_type in ("conflict", "turning_point") else "neutral",
                    transition="dissolve" if index > 0 else "fade_in",
                    music_mood=_MOOD_BY_SECTION.get(section.section_type, "neutral"),
                    sfx=["transition"] if index > 0 else [],
                    captions=True,
                    asset_requirements=[AssetRequirement(
                        description=f"{visual_type.value} illustrating: {chunk[:80]}",
                        visual_type=visual_type,
                    )],
                )
                scenes.append(scene)
                index += 1

        plan = ScenePlan(scenes=scenes)
        plan.recompute()
        logger.info("Planned %d scenes, total duration %.1fs", len(scenes), plan.total_duration_seconds)
        return plan

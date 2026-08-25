"""ScriptAgent: story.json + research.json -> script.json (spec section 19).

Generates original narration organized into sections (hook, context,
setup, conflict, development, turning point, consequences, analysis,
conclusion). Section selection is driven by the story structure so the
output isn't a single fixed template, and hooks are varied so every
video does not open with "In today's video...".
"""
from __future__ import annotations

import random

from core.logging import get_logger
from core.models import Claim, Research, Script, ScriptSection, Story, StoryStructure

logger = get_logger(__name__)

_HOOK_TEMPLATES = [
    "It started with a decision nobody expected {topic} to make.",
    "Most people only know one side of {topic}. The full story is different.",
    "By the time anyone noticed, {topic} had already changed.",
    "There's a moment that explains everything that happened to {topic} — and it's not the one you'd guess.",
    "{topic} almost didn't happen this way at all.",
]

# Section templates per structure keep narrative shape varied while
# reusing the same underlying section_type vocabulary from the spec.
_STRUCTURE_SECTIONS: dict[StoryStructure, list[str]] = {
    StoryStructure.RISE_FALL: ["hook", "context", "setup", "development", "conflict", "turning_point", "consequences", "conclusion"],
    StoryStructure.RISE_TRANSFORMATION: ["hook", "context", "setup", "turning_point", "development", "consequences", "conclusion"],
    StoryStructure.MYSTERY: ["hook", "setup", "investigation", "development", "turning_point", "analysis", "conclusion"],
    StoryStructure.INVESTIGATION: ["hook", "context", "investigation", "development", "analysis", "conclusion"],
    StoryStructure.CHRONOLOGY: ["hook", "context", "setup", "development", "development", "consequences", "conclusion"],
    StoryStructure.INVENTION: ["hook", "context", "setup", "development", "turning_point", "consequences", "conclusion"],
    StoryStructure.CONFLICT: ["hook", "context", "conflict", "development", "turning_point", "consequences", "conclusion"],
    StoryStructure.COMPETITION: ["hook", "context", "setup", "conflict", "development", "turning_point", "conclusion"],
    StoryStructure.DISASTER_INVESTIGATION: ["hook", "context", "setup", "conflict", "investigation", "analysis", "conclusion"],
    StoryStructure.BIOGRAPHY: ["hook", "context", "setup", "development", "turning_point", "consequences", "conclusion"],
    StoryStructure.TECH_EXPLANATION: ["hook", "context", "setup", "development", "analysis", "conclusion"],
    StoryStructure.SCIENTIFIC_DISCOVERY: ["hook", "context", "setup", "development", "turning_point", "analysis", "conclusion"],
    StoryStructure.BUSINESS_CASE_STUDY: ["hook", "context", "setup", "conflict", "development", "consequences", "conclusion"],
    StoryStructure.TURNING_POINT: ["hook", "context", "setup", "turning_point", "consequences", "conclusion"],
    StoryStructure.CAUSE_EFFECT: ["hook", "context", "setup", "development", "consequences", "conclusion"],
    StoryStructure.BEFORE_AFTER: ["hook", "context", "setup", "turning_point", "consequences", "conclusion"],
}


class ScriptAgent:
    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def run(self, research: Research, story: Story) -> Script:
        section_types = _STRUCTURE_SECTIONS.get(story.structures[0], _STRUCTURE_SECTIONS[StoryStructure.CHRONOLOGY])

        claims_by_index = list(research.claims)
        sections: list[ScriptSection] = []

        hook_template = self._rng.choice(_HOOK_TEMPLATES)
        for i, section_type in enumerate(section_types):
            claim_slice = claims_by_index[i::max(1, len(section_types))][:2]
            narration = self._draft_narration(section_type, research, story, claim_slice, hook_template)
            sections.append(ScriptSection(
                section_type=section_type,
                narration=narration,
                claim_ids=[c.claim_id for c in claim_slice],
            ))

        script = Script(topic=research.topic, sections=sections)
        script.recompute()
        logger.info("Generated script with %d sections (~%.0fs)", len(sections), script.estimated_duration_seconds)
        return script

    def _draft_narration(
        self, section_type: str, research: Research, story: Story, claims: list[Claim], hook_template: str
    ) -> str:
        topic = research.topic
        claim_text = " ".join(c.text for c in claims) if claims else ""

        if section_type == "hook":
            return hook_template.format(topic=topic)
        if section_type == "context":
            return f"To understand what happened, it helps to see where {topic} started. {claim_text}".strip()
        if section_type == "setup":
            return f"The pieces were already in motion before anyone recognized the pattern. {claim_text}".strip()
        if section_type == "conflict":
            return f"Not everything went smoothly. Tension built as circumstances around {topic} shifted. {claim_text}".strip()
        if section_type == "investigation":
            return f"Piecing together what really happened required looking past the official account. {claim_text}".strip()
        if section_type == "development":
            return f"From there, events unfolded in ways that reshaped what came next. {claim_text}".strip()
        if section_type == "turning_point":
            return f"Then came the moment that changed the trajectory of {topic} for good. {claim_text}".strip()
        if section_type == "consequences":
            return f"The effects rippled outward, well beyond what anyone initially expected. {claim_text}".strip()
        if section_type == "analysis":
            return f"Looking back, a few factors stand out as decisive. {claim_text}".strip()
        if section_type == "conclusion":
            return f"What {topic} leaves behind is a story that's still being understood today."
        return claim_text or f"More about {topic}."

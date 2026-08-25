"""StoryAgent: research.json -> story.json (spec section 18).

Classifies the underlying narrative into one or more StoryStructure
values using keyword/heuristic signals extracted from the research
(entities, context notes, topic text). Topic-agnostic: no category
(CEO, disaster, invention, ...) is hard-coded as the default -- the
classifier scores every structure and keeps the top matches.
"""
from __future__ import annotations

from core.logging import get_logger
from core.models import Research, Story, StoryStructure

logger = get_logger(__name__)

# Keyword signals per structure. Deliberately broad/topic-agnostic so the
# same classifier works for CEOs, disasters, inventions, science, etc.
_SIGNALS: dict[StoryStructure, list[str]] = {
    StoryStructure.RISE_FALL: ["rise", "fall", "collapse", "bankrupt", "downfall", "decline"],
    StoryStructure.RISE_TRANSFORMATION: ["transform", "pivot", "reinvent", "turnaround"],
    StoryStructure.MYSTERY: ["mystery", "unknown", "unexplained", "disappear", "vanish"],
    StoryStructure.INVESTIGATION: ["investigat", "inquiry", "uncover", "expose", "evidence"],
    StoryStructure.CHRONOLOGY: ["timeline", "history of", "over the years", "decade"],
    StoryStructure.INVENTION: ["invent", "patent", "prototype", "breakthrough device"],
    StoryStructure.CONFLICT: ["conflict", "battle", "clash", "dispute", "war"],
    StoryStructure.COMPETITION: ["competitor", "rival", "market share", "versus", "race"],
    StoryStructure.DISASTER_INVESTIGATION: ["disaster", "crash", "explosion", "failure", "catastrophe"],
    StoryStructure.BIOGRAPHY: ["life of", "born", "career", "biography"],
    StoryStructure.TECH_EXPLANATION: ["how it works", "mechanism", "engineering", "architecture"],
    StoryStructure.SCIENTIFIC_DISCOVERY: ["discover", "research team", "experiment", "hypothesis"],
    StoryStructure.BUSINESS_CASE_STUDY: ["company", "startup", "business model", "revenue", "strategy"],
    StoryStructure.TURNING_POINT: ["turning point", "pivotal", "decisive moment"],
    StoryStructure.CAUSE_EFFECT: ["caused", "led to", "resulted in", "consequence"],
    StoryStructure.BEFORE_AFTER: ["before", "after", "changed everything", "prior to"],
}


class StoryAgent:
    def run(self, research: Research) -> Story:
        haystack = " ".join(
            [research.topic.lower()]
            + [n.lower() for n in research.context_notes]
            + [c.text.lower() for c in research.claims]
            + [(e.description or "").lower() for e in research.entities]
        )

        scores: dict[StoryStructure, int] = {}
        for structure, keywords in _SIGNALS.items():
            score = sum(haystack.count(kw) for kw in keywords)
            if score > 0:
                scores[structure] = score

        if not scores:
            # Safe topic-agnostic default when no strong signal is found:
            # present as a chronology, the most neutral structure.
            structures = [StoryStructure.CHRONOLOGY]
        else:
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            structures = [s for s, _ in ranked[:2]]

        logger.info("Classified story structure(s) for %r: %s", research.topic, structures)

        themes = [e.name for e in research.entities if e.entity_type == "concept"]
        logline = f"The story of {research.topic}, told through {structures[0].value.replace('_', ' ')}."

        return Story(topic=research.topic, structures=structures, logline=logline, themes=themes)

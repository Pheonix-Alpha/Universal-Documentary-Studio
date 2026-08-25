"""MetadataAgent: script.json + scenes.json -> metadata (spec section 42).

Generates title options, a description, chapter markers (from section
boundaries), keywords, and hashtags. Never fabricates claims about the
content (no misleading titles/thumbnails per spec 41/42).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.models import Script


@dataclass
class VideoMetadata:
    title_options: list[str]
    description: str
    chapters: list[dict]
    keywords: list[str]
    hashtags: list[str]


class MetadataAgent:
    def run(self, script: Script, topic: str) -> VideoMetadata:
        title_options = [
            f"{topic}: The Full Story",
            f"How {topic} Really Happened",
            f"{topic} — What Actually Went Wrong (and Right)",
            f"The Untold Story of {topic}",
        ]

        description = (
            f"An in-depth look at {topic}, examining the key events, decisions, and consequences "
            f"that shaped the outcome. Sourced from publicly available records and reporting."
        )

        chapters = []
        cursor = 0.0
        for section in script.sections:
            word_count = len(section.narration.split())
            duration = (word_count / 150.0) * 60.0
            chapters.append({
                "title": section.section_type.replace("_", " ").title(),
                "start_seconds": round(cursor, 1),
            })
            cursor += duration

        keywords = list({topic.lower(), "documentary", "explained", "history", "analysis"})
        hashtags = [f"#{topic.replace(' ', '')}", "#documentary", "#explained"]

        return VideoMetadata(
            title_options=title_options, description=description,
            chapters=chapters, keywords=keywords, hashtags=hashtags,
        )

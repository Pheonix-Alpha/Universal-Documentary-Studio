"""Core data models shared across the whole pipeline.

These are the schemas persisted as checkpoints (research.json, script.json,
scenes.json, ...). Everything is a pydantic BaseModel so we get validation,
JSON (de)serialization, and clear error messages for free.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Project state machine
# ---------------------------------------------------------------------------

class ProjectState(str, Enum):
    CREATED = "CREATED"
    RESEARCHING = "RESEARCHING"
    RESEARCH_COMPLETE = "RESEARCH_COMPLETE"
    FACT_CHECKING = "FACT_CHECKING"
    FACT_CHECK_COMPLETE = "FACT_CHECK_COMPLETE"
    SCRIPTING = "SCRIPTING"
    SCRIPT_COMPLETE = "SCRIPT_COMPLETE"
    SCENE_PLANNING = "SCENE_PLANNING"
    SCENES_COMPLETE = "SCENES_COMPLETE"
    ASSET_GENERATION = "ASSET_GENERATION"
    ASSETS_COMPLETE = "ASSETS_COMPLETE"
    AUDIO_GENERATION = "AUDIO_GENERATION"
    AUDIO_COMPLETE = "AUDIO_COMPLETE"
    RENDERING = "RENDERING"
    RENDER_COMPLETE = "RENDER_COMPLETE"
    QA = "QA"
    QA_FAILED = "QA_FAILED"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    APPROVED = "APPROVED"
    EXPORTED = "EXPORTED"
    FAILED = "FAILED"


# Legal forward transitions. QA_FAILED / FAILED can be reached from most
# "active" states, and regeneration can move backwards intentionally, so
# those are handled separately by the StateMachine rather than listed here.
STATE_TRANSITIONS: dict[ProjectState, list[ProjectState]] = {
    ProjectState.CREATED: [ProjectState.RESEARCHING],
    ProjectState.RESEARCHING: [ProjectState.RESEARCH_COMPLETE, ProjectState.FAILED],
    ProjectState.RESEARCH_COMPLETE: [ProjectState.FACT_CHECKING],
    ProjectState.FACT_CHECKING: [ProjectState.FACT_CHECK_COMPLETE, ProjectState.FAILED],
    ProjectState.FACT_CHECK_COMPLETE: [ProjectState.SCRIPTING],
    ProjectState.SCRIPTING: [ProjectState.SCRIPT_COMPLETE, ProjectState.FAILED],
    ProjectState.SCRIPT_COMPLETE: [ProjectState.SCENE_PLANNING],
    ProjectState.SCENE_PLANNING: [ProjectState.SCENES_COMPLETE, ProjectState.FAILED],
    ProjectState.SCENES_COMPLETE: [ProjectState.ASSET_GENERATION],
    ProjectState.ASSET_GENERATION: [ProjectState.ASSETS_COMPLETE, ProjectState.FAILED],
    ProjectState.ASSETS_COMPLETE: [ProjectState.AUDIO_GENERATION],
    ProjectState.AUDIO_GENERATION: [ProjectState.AUDIO_COMPLETE, ProjectState.FAILED],
    ProjectState.AUDIO_COMPLETE: [ProjectState.RENDERING],
    ProjectState.RENDERING: [ProjectState.RENDER_COMPLETE, ProjectState.FAILED],
    ProjectState.RENDER_COMPLETE: [ProjectState.QA],
    ProjectState.QA: [ProjectState.HUMAN_REVIEW, ProjectState.QA_FAILED],
    ProjectState.QA_FAILED: [ProjectState.ASSET_GENERATION, ProjectState.SCRIPTING,
                              ProjectState.SCENE_PLANNING, ProjectState.RENDERING,
                              ProjectState.FAILED],
    ProjectState.HUMAN_REVIEW: [ProjectState.APPROVED, ProjectState.SCENE_PLANNING,
                                 ProjectState.SCRIPTING, ProjectState.ASSET_GENERATION,
                                 ProjectState.AUDIO_GENERATION, ProjectState.FAILED],
    ProjectState.APPROVED: [ProjectState.EXPORTED],
    ProjectState.EXPORTED: [],
    ProjectState.FAILED: [ProjectState.RESEARCHING, ProjectState.SCRIPTING],
}


# ---------------------------------------------------------------------------
# Sources / research
# ---------------------------------------------------------------------------

class SourceType(str, Enum):
    PRIMARY = "primary"
    OFFICIAL = "official"
    GOVERNMENT = "government"
    ACADEMIC = "academic"
    JOURNALISM = "journalism"
    REFERENCE = "reference"
    OTHER = "other"


class Source(BaseModel):
    source_id: str = Field(default_factory=lambda: _new_id("src"))
    title: str
    url: str
    publisher: Optional[str] = None
    author: Optional[str] = None
    published_date: Optional[str] = None
    retrieved_date: str = Field(default_factory=_now)
    source_type: SourceType = SourceType.OTHER
    reliability: float = 0.5

    @field_validator("reliability")
    @classmethod
    def _clamp_reliability(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class Claim(BaseModel):
    claim_id: str = Field(default_factory=lambda: _new_id("claim"))
    text: str
    source_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.5

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class ResearchEntity(BaseModel):
    name: str
    entity_type: str  # person / organization / location / date / statistic / concept
    description: Optional[str] = None


class Research(BaseModel):
    topic: str
    entities: list[ResearchEntity] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    context_notes: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Fact checking
# ---------------------------------------------------------------------------

class FactCheckStatus(str, Enum):
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


class FactCheckResult(BaseModel):
    claim_id: str
    status: FactCheckStatus
    reasons: list[str] = Field(default_factory=list)


class FactCheckReport(BaseModel):
    results: list[FactCheckResult] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now)

    @property
    def has_critical_failure(self) -> bool:
        return any(r.status == FactCheckStatus.FAILED for r in self.results)


# ---------------------------------------------------------------------------
# Story / script
# ---------------------------------------------------------------------------

class StoryStructure(str, Enum):
    RISE_FALL = "rise_fall"
    RISE_TRANSFORMATION = "rise_transformation"
    MYSTERY = "mystery"
    INVESTIGATION = "investigation"
    CHRONOLOGY = "chronology"
    INVENTION = "invention"
    CONFLICT = "conflict"
    COMPETITION = "competition"
    DISASTER_INVESTIGATION = "disaster_investigation"
    BIOGRAPHY = "biography"
    TECH_EXPLANATION = "technology_explanation"
    SCIENTIFIC_DISCOVERY = "scientific_discovery"
    BUSINESS_CASE_STUDY = "business_case_study"
    TURNING_POINT = "turning_point"
    CAUSE_EFFECT = "cause_effect"
    BEFORE_AFTER = "before_after"


class Story(BaseModel):
    topic: str
    structures: list[StoryStructure]
    logline: str
    themes: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now)


class ScriptSection(BaseModel):
    section_id: str = Field(default_factory=lambda: _new_id("sec"))
    section_type: str  # hook / context / setup / conflict / development / ...
    narration: str
    claim_ids: list[str] = Field(default_factory=list)


class Script(BaseModel):
    topic: str
    sections: list[ScriptSection] = Field(default_factory=list)
    word_count: int = 0
    estimated_duration_seconds: float = 0.0
    created_at: str = Field(default_factory=_now)

    def recompute(self) -> None:
        self.word_count = sum(len(s.narration.split()) for s in self.sections)
        # ~150 words per minute average narration pace
        self.estimated_duration_seconds = (self.word_count / 150.0) * 60.0


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------

class VisualType(str, Enum):
    REAL_MEDIA = "real_media"
    PUBLIC_DOMAIN_MEDIA = "public_domain_media"
    LICENSED_MEDIA = "licensed_media"
    GENERATED_IMAGE = "generated_image"
    GENERATED_VIDEO = "generated_video"
    IMAGE_ANIMATION = "image_animation"
    PARALLAX = "parallax"
    CHART = "chart"
    TIMELINE = "timeline"
    MAP = "map"
    DIAGRAM = "diagram"
    TECHNICAL_ANIMATION = "technical_animation"
    TEXT_ANIMATION = "text_animation"
    DOCUMENT_ANIMATION = "document_animation"
    MIXED_MEDIA = "mixed_media"


class CameraMovement(str, Enum):
    STATIC = "static"
    SLOW_PUSH_IN = "slow_push_in"
    SLOW_PULL_OUT = "slow_pull_out"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT = "tilt"
    DOLLY = "dolly"
    TRACKING = "tracking"
    ORBIT = "orbit"
    PARALLAX = "parallax"
    HANDHELD = "handheld"
    CRANE = "crane"
    ZOOM = "zoom"


class LensProfile(str, Enum):
    MM24 = "24mm"
    MM35 = "35mm"
    MM50 = "50mm"
    MM85 = "85mm"


class AssetRequirement(BaseModel):
    description: str
    visual_type: VisualType


class Scene(BaseModel):
    scene_id: str = Field(default_factory=lambda: _new_id("scene"))
    index: int
    narration: str
    duration_seconds: float
    claim_ids: list[str] = Field(default_factory=list)
    visual_objective: str = ""
    visual_type: VisualType = VisualType.IMAGE_ANIMATION
    camera_movement: CameraMovement = CameraMovement.STATIC
    lens: LensProfile = LensProfile.MM35
    composition: str = "centered"
    lighting: str = "neutral"
    transition: str = "cut"
    music_mood: str = "neutral"
    sfx: list[str] = Field(default_factory=list)
    captions: bool = True
    asset_requirements: list[AssetRequirement] = Field(default_factory=list)


class ScenePlan(BaseModel):
    scenes: list[Scene] = Field(default_factory=list)
    total_duration_seconds: float = 0.0
    created_at: str = Field(default_factory=_now)

    def recompute(self) -> None:
        self.total_duration_seconds = sum(s.duration_seconds for s in self.scenes)


# ---------------------------------------------------------------------------
# Assets / licensing
# ---------------------------------------------------------------------------

class AssetOrigin(str, Enum):
    AI_GENERATED = "ai_generated"
    HUMAN_CREATED = "human_created"
    EXTERNAL_MEDIA = "external_media"


class LicenseRecord(BaseModel):
    asset_id: str
    license: str
    commercial_use: bool
    attribution_required: bool = False
    attribution_text: Optional[str] = None


class Asset(BaseModel):
    asset_id: str = Field(default_factory=lambda: _new_id("asset"))
    scene_id: str
    visual_type: VisualType
    origin: AssetOrigin
    file_path: Optional[str] = None
    provider: Optional[str] = None
    license: Optional[LicenseRecord] = None
    metadata: dict = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now)


class AssetManifest(BaseModel):
    assets: list[Asset] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

class VoiceTrack(BaseModel):
    scene_id: str
    file_path: str
    duration_seconds: float
    voice: str = "default"
    language: str = "en"


class VoiceManifest(BaseModel):
    tracks: list[VoiceTrack] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now)


class MusicCue(BaseModel):
    scene_id: str
    mood: str
    file_path: str


class MusicManifest(BaseModel):
    cues: list[MusicCue] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Render / QA
# ---------------------------------------------------------------------------

class RenderTarget(str, Enum):
    LONG_FORM = "long_form"
    SHORT = "short"


class RenderManifestEntry(BaseModel):
    target: RenderTarget
    output_path: str
    width: int
    height: int
    fps: int
    duration_seconds: float


class RenderManifest(BaseModel):
    entries: list[RenderManifestEntry] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now)


class QAStatus(str, Enum):
    READY_FOR_REVIEW = "ready_for_human_review"
    REVIEW_REQUIRED = "review_required"
    REGENERATE = "regenerate_weak_areas"
    REJECT = "reject"


class QAIssue(BaseModel):
    scene_id: Optional[str] = None
    category: str
    severity: str  # info / warning / critical
    message: str


class QAReport(BaseModel):
    score: float
    status: QAStatus
    issues: list[QAIssue] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TTS configuration
# ---------------------------------------------------------------------------

class TTSConfig(BaseModel):
    provider: str = "mock"          # "mock" or "piper"
    model_path: str = ""            # required when provider == "piper"
    use_cuda: bool = False


class ProjectConfig(BaseModel):
    project_id: str = Field(default_factory=lambda: _new_id("proj"))
    topic: str
    language: str = "en"
    target_duration_minutes: float = 10.0
    short_count: int = 4
    visual_style: str = "documentary_realism"
    voice: str = "default"
    research_depth: str = "standard"  # quick / standard / deep
    mock_mode: bool = True
    local_gpu_enabled: bool = False
    tts: TTSConfig = Field(default_factory=TTSConfig)
    created_at: str = Field(default_factory=_now)

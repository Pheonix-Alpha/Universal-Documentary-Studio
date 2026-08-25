"""AudioAgent: scenes.json -> voice.json + music.json (+ captions) (spec 32-37).

Generates per-scene narration via VoiceEngine, cleans/normalizes it,
selects a mood-matched music cue and any requested SFX, and produces
synchronized SRT/VTT captions across the whole scene plan.
"""
from __future__ import annotations

from pathlib import Path

from adapters.tts.base import TTSRequest, VoiceEngine
from core.logging import get_logger
from core.models import MusicCue, MusicManifest, ScenePlan, VoiceManifest, VoiceTrack
from engines.captions.caption_engine import segments_from_scenes, write_srt, write_vtt
from engines.music.music_engine import find_or_synthesize_cue
from engines.sfx.sfx_engine import find_or_synthesize_sfx
from engines.voice.audio_pipeline import normalize_narration

logger = get_logger(__name__)


class AudioAgent:
    def __init__(self, voice_engine: VoiceEngine, audio_dir: str, captions_dir: str,
                 music_library_dir: str, sfx_library_dir: str, voice: str = "default", language: str = "en"):
        self.voice_engine = voice_engine
        self.audio_dir = Path(audio_dir)
        self.captions_dir = Path(captions_dir)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.captions_dir.mkdir(parents=True, exist_ok=True)
        self.music_library_dir = music_library_dir
        self.sfx_library_dir = sfx_library_dir
        self.voice = voice
        self.language = language

    def run(self, scene_plan: ScenePlan) -> tuple[VoiceManifest, MusicManifest]:
        voice_tracks: list[VoiceTrack] = []
        music_cues: list[MusicCue] = []

        for scene in scene_plan.scenes:
            raw_path = str(self.audio_dir / f"scene_{scene.index:03d}_voice_raw.wav")
            clean_path = str(self.audio_dir / f"scene_{scene.index:03d}_voice.wav")

            tts_result = self.voice_engine.synthesize(
                TTSRequest(text=scene.narration, voice=self.voice, language=self.language), raw_path,
            )
            try:
                normalize_narration(raw_path, clean_path)
                final_path = clean_path
            except Exception as exc:  # noqa: BLE001
                logger.warning("Narration normalization failed for scene %d, using raw: %s", scene.index, exc)
                final_path = raw_path

            voice_tracks.append(VoiceTrack(
                scene_id=scene.scene_id, file_path=final_path,
                duration_seconds=tts_result.duration_seconds, voice=self.voice, language=self.language,
            ))

            music_path = find_or_synthesize_cue(
                scene.music_mood, scene.duration_seconds, self.music_library_dir, str(self.audio_dir),
            )
            music_cues.append(MusicCue(scene_id=scene.scene_id, mood=scene.music_mood, file_path=music_path))

            for sfx_name in scene.sfx:
                find_or_synthesize_sfx(sfx_name, min(scene.duration_seconds, 2.0),
                                        self.sfx_library_dir, str(self.audio_dir))

        self._write_captions(scene_plan)

        return VoiceManifest(tracks=voice_tracks), MusicManifest(cues=music_cues)

    def _write_captions(self, scene_plan: ScenePlan) -> None:
        scene_texts = [(s.narration, s.duration_seconds) for s in scene_plan.scenes]
        segments = segments_from_scenes(scene_texts)
        write_srt(segments, str(self.captions_dir / "captions.srt"))
        write_vtt(segments, str(self.captions_dir / "captions.vtt"))

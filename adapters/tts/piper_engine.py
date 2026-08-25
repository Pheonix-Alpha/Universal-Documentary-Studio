"""Piper TTS VoiceEngine implementation.

Provides real local Piper text-to-speech while preserving the
provider-agnostic VoiceEngine interface used by AudioAgent.
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

from adapters.tts.base import TTSRequest, TTSResult, VoiceEngine
from core.logging import get_logger

logger = get_logger(__name__)


class PiperVoiceEngine(VoiceEngine):
    """Real Piper TTS implementation.

    Piper models are local ONNX files, so no API key is required.
    """

    model_name = "piper-tts"
    provider = "rhasspy-piper"

    def __init__(
        self,
        model_path: str,
        use_cuda: bool = False,
        python_executable: str = "python",
    ) -> None:
        self.model_path = Path(model_path)
        self.use_cuda = use_cuda
        self.python_executable = python_executable

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Piper model not found: {self.model_path}"
            )

        self.config_path = self.model_path.with_suffix(".onnx.json")

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Piper model config not found: {self.config_path}"
            )

    # ------------------------------------------------------------------
    # VoiceEngine
    # ------------------------------------------------------------------

    def synthesize(
        self,
        request: TTSRequest,
        output_path: str,
    ) -> TTSResult:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        command = [
            self.python_executable,
            "-m",
            "piper",
            "-m",
            str(self.model_path),
            "-f",
            str(output),
        ]

        if self.use_cuda:
            command.append("--cuda")

        logger.info(
            "Piper TTS: model=%s output=%s",
            self.model_path.name,
            output,
        )

        process = subprocess.run(
            command,
            input=request.text,
            text=True,
            capture_output=True,
        )

        if process.returncode != 0:
            raise RuntimeError(
                "Piper TTS synthesis failed.\n"
                f"stdout:\n{process.stdout}\n"
                f"stderr:\n{process.stderr}"
            )

        if not output.exists():
            raise RuntimeError(
                f"Piper completed without producing audio: {output}"
            )

        duration = self._get_duration(output)

        logger.info(
            "Piper TTS complete: %.2fs -> %s",
            duration,
            output,
        )

        return TTSResult(
            file_path=str(output),
            duration_seconds=duration,
            model_name=self.model_name,
            provider=self.provider,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _get_duration(path: Path) -> float:
        """Read WAV duration without requiring FFmpeg."""

        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            sample_rate = wav.getframerate()

        if sample_rate <= 0:
            raise RuntimeError(
                f"Invalid sample rate in Piper output: {path}"
            )

        return round(frames / sample_rate, 2)
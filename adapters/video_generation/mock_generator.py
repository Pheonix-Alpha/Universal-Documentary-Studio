"""Mock VideoGenerator.

Real AI video generation (SVD, Runway, Kling, ...) needs a real GPU and
is entirely optional. This mock produces an actual short MP4 clip via
FFmpeg's zoompan filter over a still image, so downstream rendering code
gets real, playable media to composite in both MOCK_MODE and CI tests.
"""
from __future__ import annotations

import subprocess

from adapters.image_generation.mock_generator import MockImageGenerator
from adapters.image_generation.base import ImageGenerationRequest
from adapters.video_generation.base import VideoGenerationRequest, VideoGenerationResult, VideoGenerator
from core.exceptions import RenderError


class MockVideoGenerator(VideoGenerator):
    model_name = "mock-video-v1"
    provider = "mock"

    def generate(self, request: VideoGenerationRequest, output_path: str) -> VideoGenerationResult:
        source_image = request.source_image_path
        if source_image is None:
            source_image = output_path.rsplit(".", 1)[0] + "_source.png"
            MockImageGenerator().generate(
                ImageGenerationRequest(prompt=request.prompt, width=request.width, height=request.height),
                output_path=source_image,
            )

        frames = max(1, int(request.duration_seconds * request.fps))
        zoom_expr = f"zoom+0.0008"
        vf = (
            f"scale={request.width * 2}:{request.height * 2},"
            f"zoompan=z='{zoom_expr}':d={frames}:s={request.width}x{request.height}:fps={request.fps}"
        )
        cmd = [
            "ffmpeg", "-y", "-loop", "1", "-i", source_image,
            "-vf", vf, "-t", str(request.duration_seconds),
            "-pix_fmt", "yuv420p", output_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RenderError(f"mock video generation (ffmpeg) failed: {proc.stderr[-2000:]}")

        return VideoGenerationResult(
            file_path=output_path,
            model_name=self.model_name,
            provider=self.provider,
            metadata={"prompt": request.prompt, "source_image": source_image, "frames": frames},
        )

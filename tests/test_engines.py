from __future__ import annotations

import os
import wave

from PIL import Image

from core.models import CameraMovement
from engines.animation.ken_burns import animate_image
from engines.captions.caption_engine import CaptionSegment, segments_from_scenes, write_srt, write_vtt
from engines.charts.chart_engine import render_bar_chart, render_line_chart
from engines.rendering.ffmpeg_renderer import get_video_info


def _make_source_image(path: str, size=(320, 180)):
    img = Image.new("RGB", size, (40, 60, 90))
    img.save(path)


def test_animate_image_produces_valid_video_for_every_movement(tmp_path):
    src = str(tmp_path / "src.png")
    _make_source_image(src)
    for movement in CameraMovement:
        out = str(tmp_path / f"out_{movement.value}.mp4")
        animate_image(src, out, duration_seconds=0.5, movement=movement, width=160, height=90, fps=10)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
        info = get_video_info(out)
        assert info.get("width") == "160"
        assert info.get("height") == "90"


def test_segments_from_scenes_computes_cumulative_timing():
    segments = segments_from_scenes([("first", 2.0), ("second", 3.0)])
    assert segments[0].start_seconds == 0.0
    assert segments[0].end_seconds == 2.0
    assert segments[1].start_seconds == 2.0
    assert segments[1].end_seconds == 5.0


def test_write_srt_and_vtt_produce_valid_files(tmp_path):
    segments = [CaptionSegment(text="Hello world", start_seconds=0.0, end_seconds=1.5)]
    srt_path = str(tmp_path / "out.srt")
    vtt_path = str(tmp_path / "out.vtt")
    write_srt(segments, srt_path)
    write_vtt(segments, vtt_path)

    srt_content = open(srt_path).read()
    assert "Hello world" in srt_content
    assert "00:00:00,000 --> 00:00:01,500" in srt_content

    vtt_content = open(vtt_path).read()
    assert vtt_content.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.500" in vtt_content


def test_render_bar_and_line_chart_produce_image_files(tmp_path):
    bar_path = str(tmp_path / "bar.png")
    line_path = str(tmp_path / "line.png")
    render_bar_chart(["A", "B", "C"], [1, 2, 3], "Test Bar", bar_path, width_px=400, height_px=300)
    render_line_chart(["A", "B", "C"], [1, 2, 3], "Test Line", line_path, width_px=400, height_px=300)
    assert os.path.exists(bar_path) and os.path.getsize(bar_path) > 0
    assert os.path.exists(line_path) and os.path.getsize(line_path) > 0

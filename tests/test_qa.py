from __future__ import annotations

from core.models import (
    Asset, AssetOrigin, Claim, LicenseRecord, QAStatus, Source, VisualType,
)
from engines.captions.caption_engine import CaptionSegment
from qa.asset_checker import check_assets
from qa.caption_checker import check_captions
from qa.license_checker import check_licenses
from qa.quality_score import build_report, score_issues, status_for_score
from qa.source_checker import check_sources


def test_source_checker_flags_unsourced_claim():
    claims = [Claim(text="Unsourced claim", source_ids=[])]
    issues = check_sources(claims, sources=[])
    assert len(issues) == 1
    assert issues[0].category == "missing_source"


def test_source_checker_passes_well_sourced_claim():
    source = Source(title="T", url="https://example.com")
    claim = Claim(text="Sourced claim", source_ids=[source.source_id], confidence=0.9)
    issues = check_sources([claim], sources=[source])
    assert issues == []


def test_license_checker_flags_unknown_license_external_media():
    asset = Asset(scene_id="s1", visual_type=VisualType.REAL_MEDIA, origin=AssetOrigin.EXTERNAL_MEDIA,
                   file_path=None, license=None)
    issues = check_licenses([asset])
    assert len(issues) == 1
    assert issues[0].category == "unknown_license"


def test_license_checker_passes_licensed_external_media():
    asset = Asset(scene_id="s1", visual_type=VisualType.REAL_MEDIA, origin=AssetOrigin.EXTERNAL_MEDIA,
                   file_path=None, license=LicenseRecord(asset_id="a1", license="CC-BY-4.0", commercial_use=True))
    issues = check_licenses([asset])
    assert issues == []


def test_license_checker_ignores_ai_generated_assets():
    asset = Asset(scene_id="s1", visual_type=VisualType.GENERATED_IMAGE, origin=AssetOrigin.AI_GENERATED,
                   file_path=None, license=None)
    issues = check_licenses([asset])
    assert issues == []


def test_asset_checker_flags_missing_file(tmp_path):
    asset = Asset(scene_id="s1", visual_type=VisualType.IMAGE_ANIMATION, origin=AssetOrigin.AI_GENERATED,
                   file_path=str(tmp_path / "does_not_exist.mp4"))
    issues = check_assets([asset])
    assert any(i.category == "missing_asset" for i in issues)


def test_asset_checker_flags_excessive_reuse(tmp_path):
    f = tmp_path / "reused.mp4"
    f.write_bytes(b"data")
    assets = [
        Asset(scene_id=f"s{i}", visual_type=VisualType.IMAGE_ANIMATION, origin=AssetOrigin.AI_GENERATED,
              file_path=str(f), metadata={"prompt": "same prompt"})
        for i in range(5)
    ]
    issues = check_assets(assets, max_reuse=3)
    assert any(i.category == "visual_reuse" for i in issues)


def test_caption_checker_flags_zero_duration_segment():
    segments = [CaptionSegment(text="x", start_seconds=1.0, end_seconds=1.0)]
    issues = check_captions(segments, total_duration_seconds=1.0)
    assert any(i.category == "caption_timing" for i in issues)


def test_caption_checker_passes_well_formed_segments():
    segments = [
        CaptionSegment(text="a", start_seconds=0.0, end_seconds=1.0),
        CaptionSegment(text="b", start_seconds=1.0, end_seconds=2.0),
    ]
    issues = check_captions(segments, total_duration_seconds=2.0)
    assert issues == []


def test_quality_score_thresholds():
    assert status_for_score(95) == QAStatus.READY_FOR_REVIEW
    assert status_for_score(85) == QAStatus.REVIEW_REQUIRED
    assert status_for_score(75) == QAStatus.REGENERATE
    assert status_for_score(50) == QAStatus.REJECT


def test_build_report_aggregates_issues_into_score():
    from core.models import QAIssue

    issues = [QAIssue(category="x", severity="critical", message="bad")]
    report = build_report(issues)
    assert report.score == 80.0
    assert report.status == QAStatus.REVIEW_REQUIRED


def test_build_report_no_issues_is_perfect_score():
    report = build_report([])
    assert report.score == 100.0
    assert report.status == QAStatus.READY_FOR_REVIEW

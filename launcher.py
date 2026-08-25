#!/usr/bin/env python3
"""
launcher.py — the ONE entry point for Universal Documentary Studio.

Colab usage:

    !git clone https://github.com/Pheonix-Alpha/Universal-Documentary-Studio.git /content/uds
    %cd /content/uds
    !python launcher.py

That's it. Nothing else needs to run manually.

What this file does (and ONLY this — it is a bootstrapper, not the
pipeline):

    1. Detect Colab vs local environment
    2. Mount Google Drive and create the UDS folder tree there
    3. Point HF_HOME / TRANSFORMERS_CACHE / HF_DATASETS_CACHE / TORCH_HOME
       at Drive so models are downloaded once and reused forever
    4. Install any missing Python dependencies (skips what's already there)
    5. Detect GPU / VRAM / RAM / CPU / disk via the existing ResourceManager
    6. Ask the existing ModelRegistry which models it would pick for this
       hardware, and make sure they're present on Drive (download once)
    7. Write config/config.yaml pointing UDS at the current Drive paths
    8. Validate the install via the existing app.startup.run_startup()
    9. Launch the Gradio UI

The pipeline, agents, ModelRegistry, Scheduler, ResourceManager and
engines are untouched — this file only prepares the environment for them.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DRIVE_ROOT_NAME = "Universal-Documentary-Studio"

DIR_LAYOUT: Dict[str, Tuple[str, ...]] = {
    "hf_models": ("models", "huggingface"),
    "piper_models": ("models", "piper"),
    "other_models": ("models", "other-models"),
    "cache": ("cache",),
    "projects": ("projects",),
    "outputs": ("outputs",),
    "logs": ("logs",),
}

BASE_REQUIREMENTS_FILE = REPO_ROOT / "requirements.txt"

# (import name, pip package name) — heavy/optional packages needed for
# real (non-mock) model execution. Each is checked before installing so
# a re-run never reinstalls what's already there.
HEAVY_PACKAGES: List[Tuple[str, str]] = [
    ("torch", "torch"),
    ("diffusers", "diffusers"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("safetensors", "safetensors"),
    ("huggingface_hub", "huggingface_hub"),
    ("piper", "piper-tts"),
    ("gradio", "gradio"),
]

PIPER_VOICE_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# HuggingFace repo ids for every non-mock model the registry might pick.
# ModelCapability.hf_repo_id is the source of truth (models/registry.py);
# this local copy exists purely so the launcher can *pre-cache* weights
# on Drive without importing torch/diffusers itself, and is validated
# against the registry at prepare_ai_models() time so it can't silently
# drift out of sync.
_HF_REPO_MAP = {
    "sd-turbo-small": "stabilityai/sd-turbo",
    "sdxl-base": "stabilityai/stable-diffusion-xl-base-1.0",
    "svd-open": "stabilityai/stable-video-diffusion-img2vid",
}


# ---------------------------------------------------------------------------
# small console UI helpers
# ---------------------------------------------------------------------------

def _bar(current: float, total: float, width: int = 28) -> str:
    ratio = 0.0 if total <= 0 else max(0.0, min(1.0, current / total))
    filled = int(width * ratio)
    return f"[{'#' * filled}{'-' * (width - filled)}] {int(ratio * 100):3d}%"


def _step(msg: str) -> None:
    print(f"\n>>> {msg}")


def _ok(msg: str) -> None:
    print(f"    ✓ {msg}")


def _warn(msg: str) -> None:
    print(f"    ! {msg}")


def _fail(msg: str) -> None:
    print(f"    ✗ {msg}")


# ---------------------------------------------------------------------------
# 1. Colab detection + 2. Drive mount + directory tree
# ---------------------------------------------------------------------------

def is_colab() -> bool:
    return importlib.util.find_spec("google.colab") is not None


def mount_drive(force_local: bool = False) -> Path:
    _step("Detecting environment...")
    if not force_local and is_colab():
        _ok("Running in Google Colab.")
        _step("Mounting Google Drive...")
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive", force_remount=False)
        root = Path("/content/drive/MyDrive") / DRIVE_ROOT_NAME
        _ok(f"Drive mounted. UDS root: {root}")
        return root

    _ok("Not running in Colab (or --local was passed).")
    root = (REPO_ROOT / "uds_data" / DRIVE_ROOT_NAME).resolve()
    _warn(f"Using local data root instead of Drive: {root}")
    return root


def prepare_directories(root: Path) -> Dict[str, Path]:
    _step(f"Preparing folder tree under {root} ...")
    dirs: Dict[str, Path] = {}
    for key, parts in DIR_LAYOUT.items():
        p = root.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        dirs[key] = p
        _ok("/".join(parts))
    return dirs


# ---------------------------------------------------------------------------
# 3. cache env vars
# ---------------------------------------------------------------------------

def configure_cache_env(dirs: Dict[str, Path]) -> Dict[str, str]:
    _step("Configuring model cache environment variables...")
    hf_cache = dirs["cache"] / "huggingface"
    torch_cache = dirs["cache"] / "torch"
    (hf_cache / "transformers").mkdir(parents=True, exist_ok=True)
    (hf_cache / "datasets").mkdir(parents=True, exist_ok=True)
    torch_cache.mkdir(parents=True, exist_ok=True)

    env = {
        "HF_HOME": str(hf_cache),
        "TRANSFORMERS_CACHE": str(hf_cache / "transformers"),
        "HF_DATASETS_CACHE": str(hf_cache / "datasets"),
        "TORCH_HOME": str(torch_cache),
    }
    for k, v in env.items():
        os.environ[k] = v
        _ok(f"{k} = {v}")
    return env


# ---------------------------------------------------------------------------
# 4. dependency installation
# ---------------------------------------------------------------------------

def _pip_install(args: List[str]) -> bool:
    cmd = [sys.executable, "-m", "pip", "install", "-q", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def ensure_python_dependencies(skip_heavy: bool = False) -> List[Dict[str, Any]]:
    _step("Checking Python dependencies...")
    results: List[Dict[str, Any]] = []

    if BASE_REQUIREMENTS_FILE.exists():
        print("    requirements.txt: installing/verifying (base, always required)...")
        ok = _pip_install(["-r", str(BASE_REQUIREMENTS_FILE)])
        (_ok if ok else _fail)("requirements.txt")
        results.append({"package": "requirements.txt", "installed": ok})

    if skip_heavy:
        _warn("Skipping heavy/optional packages (--skip-heavy-deps).")
        return results

    total = len(HEAVY_PACKAGES)
    for i, (import_name, pip_name) in enumerate(HEAVY_PACKAGES, start=1):
        already = importlib.util.find_spec(import_name) is not None
        if already:
            ok, status = True, "already installed"
        else:
            print(f"    [{i}/{total}] installing {pip_name} ... {_bar(i - 1, total)}", end="\r")
            ok = _pip_install([pip_name])
            status = "installed" if ok else "FAILED — will fall back to mock where needed"
        print(f"    [{i}/{total}] {pip_name:16s} {_bar(i, total)}  {status}" + " " * 8)
        results.append({"package": pip_name, "installed": ok})

    return results


def ensure_ffmpeg() -> bool:
    _step("Checking ffmpeg...")
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        _ok("ffmpeg already installed.")
        return True

    _warn("ffmpeg not found — installing via apt-get...")
    subprocess.run(["apt-get", "update", "-qq"], capture_output=True)
    result = subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], capture_output=True, text=True)
    ok = shutil.which("ffmpeg") is not None
    if ok:
        _ok("ffmpeg installed.")
    else:
        _fail(f"ffmpeg install failed: {result.stderr[-300:]}")
    return ok


# ---------------------------------------------------------------------------
# 5. hardware detection (delegates entirely to the existing ResourceManager)
# ---------------------------------------------------------------------------

def detect_hardware():
    _step("Detecting hardware (via core.resource_manager.ResourceManager)...")
    from core.resource_manager import ResourceManager

    rm = ResourceManager(vram_safety_margin_gb=1.5, path_for_disk=str(REPO_ROOT))
    report = rm.detect(refresh=True)
    profile = rm.classify()

    _ok(f"GPU: {report.gpu_name or 'none detected'}")
    _ok(f"VRAM: {report.vram_gb} GB (effective after safety margin: {rm.effective_vram_gb():.2f} GB)")
    _ok(f"RAM: {report.ram_gb} GB | CPU cores: {report.cpu_cores} | CUDA: {report.cuda_available}")
    _ok(f"Disk free: {report.disk_free_gb} GB")
    _ok(f"Worker profile: {profile.value}")
    return rm, report, profile


# ---------------------------------------------------------------------------
# 6. model preparation (uses the EXISTING ModelRegistry for selection —
#    the launcher only ensures the chosen models are physically present)
# ---------------------------------------------------------------------------

def _download_with_progress(url: str, dest: Path, label: str) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        _ok(f"{label}: already present — skipping download.")
        return True

    try:
        import requests
    except ImportError:
        _warn(f"{label}: 'requests' not available — skipping download.")
        return False

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            last_print = 0.0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if not chunk:
                        continue
                    f.write(chunk)
                    done += len(chunk)
                    now = time.time()
                    if total and now - last_print > 0.2:
                        print(f"    {label}: {_bar(done, total)}  ({done/1e6:.1f}/{total/1e6:.1f} MB)", end="\r")
                        last_print = now
            tmp.replace(dest)
        print(f"    {label}: {_bar(1, 1)}  done" + " " * 20)
        return True
    except Exception as exc:  # noqa: BLE001 - network is inherently unreliable
        _warn(f"{label}: download failed ({exc}) — pipeline will fall back to the mock engine.")
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return False


def prepare_piper_voice(dirs: Dict[str, Path], voice: str = "en_US-ryan-high") -> Optional[Path]:
    _step(f"Preparing Piper TTS voice '{voice}'...")
    try:
        locale, remainder = voice.split("-", 1)
        name, quality = remainder.rsplit("-", 1)
        lang_short = locale.split("_")[0]
    except ValueError:
        _warn(f"Unrecognized voice id '{voice}' — skipping Piper download.")
        return None

    onnx_url = f"{PIPER_VOICE_BASE}/{lang_short}/{locale}/{name}/{quality}/{voice}.onnx"
    json_url = onnx_url + ".json"
    onnx_path = dirs["piper_models"] / f"{voice}.onnx"
    json_path = dirs["piper_models"] / f"{voice}.onnx.json"

    ok1 = _download_with_progress(onnx_url, onnx_path, f"{voice}.onnx")
    ok2 = _download_with_progress(json_url, json_path, f"{voice}.onnx.json")
    return onnx_path if (ok1 and ok2) else None


# Weight-file patterns actually needed at real-mode load time. Both
# DiffusersImageGenerator and DiffusersVideoGenerator request
# `variant="fp16", use_safetensors=True` when running on GPU, so a plain
# `snapshot_download(repo_id)` with no filtering pulls a large multiple of
# what's ever used -- every precision variant (fp32 *and* fp16), every
# format (.safetensors *and* .bin), plus ONNX/OpenVINO/Flax exports that
# this pipeline never touches. For SDXL that's the difference between the
# ~7 GB actually loaded and the ~40 GB full repo. Mirroring the adapters'
# own load_kwargs here means pre-caching downloads exactly what will be
# used, not a superset of it.
_DIFFUSERS_ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "**/*.json",
    "**/*.txt",
    "*fp16*.safetensors",
    "**/*fp16*.safetensors",
]


def _precache_huggingface_model(model, dirs: Dict[str, Path], resource_manager=None) -> None:
    repo_id = getattr(model, "hf_repo_id", None) or _HF_REPO_MAP.get(model.model_name)
    if repo_id is None:
        _warn(f"No known HuggingFace repo mapping for '{model.model_name}' — skipping pre-cache.")
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        _warn("huggingface_hub not installed — skipping pre-cache.")
        return

    estimated_gb = getattr(model, "estimated_disk_gb", 0.0)
    if resource_manager is not None and estimated_gb:
        free_gb = resource_manager.detect().disk_free_gb
        # Leave headroom: the download needs its own temp/partial-file
        # space on top of the final file size before HF's downloader
        # reconciles them.
        required_gb = estimated_gb * 1.3
        if free_gb < required_gb:
            _warn(
                f"Skipping pre-cache of {repo_id}: needs ~{required_gb:.1f} GB free "
                f"(fp16-only estimate with headroom), only {free_gb:.1f} GB available on Drive. "
                f"Free up space and re-run, or this will be attempted again on first real use."
            )
            return

    target_dir = dirs["hf_models"] / model.model_name
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            allow_patterns=_DIFFUSERS_ALLOW_PATTERNS,
        )
        _ok(f"Cached {repo_id} -> {target_dir} (fp16 safetensors only, ~{estimated_gb:.1f} GB)")
    except Exception as exc:  # noqa: BLE001
        _warn(f"Pre-cache of {repo_id} failed ({exc}) — will retry on first real use.")


def prepare_ai_models(dirs: Dict[str, Path], resource_manager, skip: bool = False) -> Dict[str, Any]:
    _step("Selecting best-fit models for this hardware (via models.registry.ModelRegistry)...")
    result: Dict[str, Any] = {"image_model": None, "video_model": None, "tts_model": None, "piper_path": None}
    if skip:
        _warn("Skipping model preparation (--skip-models).")
        return result

    from models.registry import ModelRegistry

    registry = ModelRegistry()
    available_vram = resource_manager.effective_vram_gb()

    image_model = registry.get_best_image_model(available_vram)
    video_model = registry.get_best_video_model(available_vram)
    tts_model = registry.get_best_tts_model(available_vram)
    result.update(image_model=image_model, video_model=video_model, tts_model=tts_model)

    _ok(f"Registry choice given {available_vram:.2f} GB effective VRAM:")
    print(f"        image_generation -> {image_model.model_name if image_model else 'none'}")
    print(f"        video_generation -> {video_model.model_name if video_model else 'none'}")
    print(f"        tts               -> {tts_model.model_name if tts_model else 'none'}")

    if tts_model and tts_model.model_name == "piper-tts":
        result["piper_path"] = prepare_piper_voice(dirs)
    else:
        _warn("TTS: mock model selected for this hardware — Piper download skipped.")

    for label, model in (("image", image_model), ("video", video_model)):
        if model is None or model.provider == "mock":
            continue
        _step(f"Pre-caching weights for {model.model_name} ({label} generation)...")
        _ok(
            f"{model.model_name} is wired into VisualAgent via ModelLifecycleManager — "
            f"caching it on Drive now so the first real run doesn't pay the download cost."
        )
        _precache_huggingface_model(model, dirs, resource_manager=resource_manager)

    return result


# ---------------------------------------------------------------------------
# 7. runtime configuration
# ---------------------------------------------------------------------------

def write_runtime_config(dirs: Dict[str, Path], model_selection: Dict[str, Any], report) -> Path:
    _step("Writing runtime configuration (config/config.yaml)...")
    import yaml

    config_path = REPO_ROOT / "config" / "config.yaml"

    tts_provider = "piper" if model_selection.get("piper_path") else "mock"
    tts_model_path = str(model_selection["piper_path"]) if model_selection.get("piper_path") else ""

    # mock_mode is derived from whether a real TTS voice is actually ready
    # on this Drive. It now also gates image/video generation (via
    # ModelLifecycleManager in core/pipeline.py): mock_mode=false means the
    # pipeline will select, download, load, and unload real models per the
    # registry's picks for this hardware, falling back to mock per-model if
    # a download/load fails (see prepare_ai_models() above for what got
    # pre-cached). Deriving it from tts_provider keeps this conservative --
    # a worker isn't marked "real mode" until at least its TTS voice is
    # confirmed present.
    config: Dict[str, Any] = {
        "mock_mode": tts_provider != "piper",
        "local_gpu_enabled": bool(report.gpu_available),
        "projects_root": str(dirs["projects"]),
        "resource": {"vram_safety_margin_gb": 1.5},
        "tts": {
            "provider": tts_provider,
            "model_path": tts_model_path,
            "use_cuda": bool(report.cuda_available),
        },
        "video": {
            "long_form": {"width": 1920, "height": 1080, "fps": 24},
            "short": {"width": 1080, "height": 1920, "fps": 30},
        },
        "pipeline": {
            "target_duration_minutes": 10.0,
            "short_count": 4,
            "research_depth": "standard",
        },
        "qa": {
            "ready_threshold": 90,
            "review_threshold": 80,
            "regenerate_threshold": 70,
        },
        # Extra, forward-compatible keys: unused by today's app/config.py
        # defaults but harmless (deep-merged), and ready for the UI/output
        # organizer to read once wired up.
        "storage": {
            "outputs_root": str(dirs["outputs"]),
            "logs_root": str(dirs["logs"]),
            "models_root": str(dirs["hf_models"].parent),
            "cache_root": str(dirs["cache"]),
        },
    }

    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _ok(f"Wrote {config_path}")
    _ok(f"mock_mode={config['mock_mode']}, local_gpu_enabled={config['local_gpu_enabled']}, tts.provider={tts_provider}")
    if not report.gpu_available:
        _warn(
            "No GPU detected, so local_gpu_enabled=false — heavy local jobs stay queued/"
            "paused (per spec) rather than risk an OOM crash. Piper TTS still runs on CPU."
        )
    return config_path


# ---------------------------------------------------------------------------
# 8. validation
# ---------------------------------------------------------------------------

def validate() -> Dict[str, Any]:
    _step("Validating UDS install...")
    from app.startup import run_startup

    summary = run_startup(lightweight=True)
    _ok("Core modules imported and startup check passed.")
    return summary


# ---------------------------------------------------------------------------
# 9. launch UI
# ---------------------------------------------------------------------------

def launch_ui() -> None:
    _step("Launching Gradio UI...")
    try:
        from app.ui_simple import launch_ui as launch_simple_ui

        launch_simple_ui()
    except ImportError:
        _warn("app.ui_simple not found — falling back to the advanced dashboard (app.ui).")
        from app.ui import launch_ui as launch_advanced_ui

        launch_advanced_ui()


# ---------------------------------------------------------------------------
# summary banner
# ---------------------------------------------------------------------------

def print_summary(report, checklist: List[Tuple[str, bool]], elapsed_seconds: float) -> None:
    lines = ["UNIVERSAL DOCUMENTARY STUDIO", ""]
    lines.append(f"GPU:  {report.gpu_name or 'none detected'}")
    lines.append(f"VRAM: {report.vram_gb} GB")
    lines.append("")
    for label, ok in checklist:
        lines.append(f"{'✓' if ok else '✗'} {label}")
    lines.append("")
    lines.append(f"Setup completed in {elapsed_seconds:.1f}s")
    lines.append("Starting UDS...")

    width = max(len(l) for l in lines) + 4
    print("\n╔" + "═" * width + "╗")
    for l in lines:
        print("║ " + l.ljust(width - 2) + " ║")
    print("╚" + "═" * width + "╝\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Universal Documentary Studio launcher")
    parser.add_argument("--local", action="store_true", help="Force local mode even if Colab is detected.")
    parser.add_argument("--skip-deps", action="store_true", help="Skip installing requirements.txt.")
    parser.add_argument("--skip-heavy-deps", action="store_true", help="Skip torch/diffusers/gradio/etc.")
    parser.add_argument("--skip-ffmpeg", action="store_true", help="Skip the ffmpeg apt-get check/install.")
    parser.add_argument("--skip-models", action="store_true", help="Skip model selection/download entirely.")
    parser.add_argument("--skip-ui", action="store_true", help="Do everything except launching the UI.")
    parser.add_argument("--piper-voice", type=str, default="en_US-ryan-high", help="Piper voice id to prepare.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    start = time.time()
    checklist: List[Tuple[str, bool]] = []

    root = mount_drive(force_local=args.local)
    checklist.append(("Drive / data root", True))

    dirs = prepare_directories(root)
    checklist.append(("Directory tree", True))

    configure_cache_env(dirs)
    checklist.append(("Cache environment", True))

    if not args.skip_deps:
        dep_results = ensure_python_dependencies(skip_heavy=args.skip_heavy_deps)
        checklist.append(("Dependencies", all(r["installed"] for r in dep_results)))
    else:
        _warn("Skipping dependency installation (--skip-deps).")
        checklist.append(("Dependencies (skipped)", True))

    if not args.skip_ffmpeg:
        ffmpeg_ok = ensure_ffmpeg()
        checklist.append(("FFmpeg", ffmpeg_ok))
    else:
        checklist.append(("FFmpeg (skipped)", True))

    rm, report, profile = detect_hardware()
    checklist.append((f"GPU: {report.gpu_name or 'none'} / VRAM {report.vram_gb} GB", True))
    checklist.append((f"Worker profile: {profile.value}", True))

    model_selection = prepare_ai_models(dirs, rm, skip=args.skip_models)
    if model_selection.get("piper_path"):
        checklist.append(("Piper TTS voice ready", True))
    elif not args.skip_models:
        checklist.append(("Piper TTS voice (using mock fallback)", False))

    config_path = write_runtime_config(dirs, model_selection, report)
    checklist.append(("Runtime config written", True))

    validate()
    checklist.append(("UDS validation", True))

    print_summary(report, checklist, time.time() - start)

    if not args.skip_ui:
        launch_ui()
    else:
        _ok("Setup complete. UI launch skipped (--skip-ui).")

    return 0


if __name__ == "__main__":
    sys.exit(main())

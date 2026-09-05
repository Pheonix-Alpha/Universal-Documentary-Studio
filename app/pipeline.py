"""
Video Pipeline - Models download on workers, not main
"""

from typing import Generator, List, Dict, Any, Optional
import time
import queue
import threading
from app import (
    production_bible,
    scene_orchestrator,
    consistency_manager,
    worker_client,
    video_models,
    model_manager,
    model_selector,
    scene_analyzer,
    script_enhancer,
    cache_manager,
)


def _format_llm_progress(event):
    """
    Convert an LLM progress event into a human-readable terminal line.
    """

    event_type = event.get("type")

    if event_type == "llm_start":

        max_tokens = event.get("max_tokens", 0)

        return (
            "🧠 LOCAL LLM STARTED\n"
            f"   ├─ Model: {event.get('model', 'Qwen')}\n"
            f"   ├─ Token budget: {max_tokens}\n"
            "   └─ Generating response..."
        )

    if event_type == "llm_progress":

        tokens = event.get("generated_tokens", 0)
        max_tokens = event.get("max_tokens", 1)
        percentage = event.get("percentage", 0)

        speed = event.get("tokens_per_second", 0)
        elapsed = event.get("elapsed", 0)
        eta = event.get("eta_seconds", 0)

        return (
            f"🧠 Qwen generating... "
            f"{percentage}%\n"
            f"   ├─ Tokens: {tokens} / {max_tokens}\n"
            f"   ├─ Speed: {speed:.1f} tokens/sec\n"
            f"   ├─ Elapsed: {elapsed:.0f}s\n"
            f"   └─ Estimated remaining: {eta:.0f}s"
        )

    if event_type == "llm_complete":

        tokens = event.get("generated_tokens", 0)
        elapsed = event.get("elapsed", 0)

        return (
            "✅ LOCAL LLM COMPLETE\n"
            f"   ├─ Generated: {tokens} tokens\n"
            f"   └─ Time: {elapsed:.1f}s"
        )

    return str(event)


def run_video_pipeline(
    story: str,
    model_id: str = "auto",
    duration_per_scene: int = 4,
    auto_download: bool = True,
    fps: int = 24,
    width: int = 576,
    height: int = 320,
    quality_preference: str = "auto",
) -> Generator[Dict[str, Any], None, None]:
    """
    Run complete video pipeline.

    model_id: either an explicit model id from model_manager.MODEL_REGISTRY,
    or "auto" (default, recommended) to let model_selector pick the best
    model for the job and the connected worker's resources.

    Models download on workers, not main -- the main Colab only ever
    supervises: it picks the model and tells the worker which one to use.
    """

    # ---- Stage 1: Validate ----
    if not story or len(story.strip()) < 20:
        yield {
            "stage": "error",
            "pct": 0,
            "log": "⚠️ Please enter a longer story (minimum 20 characters)",
            "gallery": [],
            "video": None,
            "model_status": {},
        }
        return

    # ---- Stage 2: Check Workers ----
    has_workers = worker_client.is_any_connected()

    if not has_workers:
        yield {
            "stage": "warning",
            "pct": 5,
            "log": "⚠️ No GPU workers connected. Video generation may be slow or unavailable.",
            "gallery": [],
            "video": None,
            "model_status": {},
        }

    # ---- Stage 2b: Make sure the local "brain" LLM is ready ----
    # First run on a fresh Colab downloads ~15GB and 4-bit-quantizes it into
    # VRAM, which can take a few minutes -- stream real progress instead of
    # leaving the UI looking stuck during "Creating production bible...".
    # Subsequent runs in the same session are instant (model_manager caches
    # it in _loaded_models).
    if not model_manager.is_installed(model_manager.LOCAL_BRAIN_MODEL_ID):
        for pct, msg in model_manager.smart_download_model(
            model_manager.LOCAL_BRAIN_MODEL_ID
        ):
            yield {
                "stage": "brain_loading",
                "pct": 6 + (pct / 100) * 3,  # 6-9%
                "log": f"🧠 {msg}",
                "gallery": [],
                "video": None,
                "model_status": {},
            }

    # ---- Stage 3: Generate Production Bible ----

    # ========================================================
    # CHECK DRIVE CACHE FIRST
    # ========================================================

    cached_brain = cache_manager.load_brain_result(story)

    if cached_brain is not None:

        bible = cached_brain["bible"]

        cached_script = cached_brain.get("enhanced_script", "")

        yield {
            "stage": "cache",
            "pct": 20,
            "log": (
                "⚡ CACHE HIT\n"
                "   ├─ Production Bible loaded from Drive\n"
                "   ├─ Enhanced script loaded from Drive\n"
                "   └─ Skipping local Qwen processing"
            ),
            "gallery": [],
            "video": None,
            "model_status": {},
        }

    else:

        yield {
            "stage": "bible",
            "pct": 10,
            "log": "📖 Creating production bible (local brain)...",
            "gallery": [],
            "video": None,
            "model_status": {},
        }

        # Queue for receiving live LLM progress events
        llm_events = queue.Queue()

        def llm_progress_callback(event):
            llm_events.put(event)

        result_holder = {}
        error_holder = {}

        def _run_brain():

            try:

                # ---------------------------------------------
                # Generate Production Bible
                # ---------------------------------------------

                bible = production_bible.generate_production_bible(
                    story, progress_callback=llm_progress_callback
                )

                # ---------------------------------------------
                # Enhance Script
                # ---------------------------------------------

                enhanced_story = script_enhancer.enhance_script(
                    story, bible, progress_callback=llm_progress_callback
                )

                # ---------------------------------------------
                # Re-analyze scenes if script changed
                # ---------------------------------------------

                if enhanced_story and enhanced_story.strip() != story.strip():

                    bible.scenes = production_bible._enrich_scenes_with_bible(
                        scene_analyzer.analyze_story(
                            enhanced_story, progress_callback=llm_progress_callback
                        ),
                        bible,
                    )

                result_holder["bible"] = bible
                result_holder["enhanced_script"] = enhanced_story

            except Exception as exc:

                error_holder["error"] = exc

        # ---------------------------------------------
        # Run Qwen in background
        # ---------------------------------------------

        brain_thread = threading.Thread(target=_run_brain, daemon=True)

        brain_thread.start()

        # ---------------------------------------------
        # Live Qwen progress
        # ---------------------------------------------

        while brain_thread.is_alive():

            try:

                event = llm_events.get(timeout=0.2)

                percentage = event.get("percentage", 0)

                yield {
                    "stage": "brain_generating",
                    "pct": 10 + (percentage * 0.10),
                    "log": _format_llm_progress(event),
                    "gallery": [],
                    "video": None,
                    "model_status": {},
                }

            except queue.Empty:

                continue

        # ---------------------------------------------
        # Send remaining events
        # ---------------------------------------------

        while not llm_events.empty():

            event = llm_events.get()

            percentage = event.get("percentage", 0)

            yield {
                "stage": "brain_generating",
                "pct": 10 + (percentage * 0.10),
                "log": _format_llm_progress(event),
                "gallery": [],
                "video": None,
                "model_status": {},
            }

        brain_thread.join()

        # ---------------------------------------------
        # Handle Qwen failure
        # ---------------------------------------------

        if "error" in error_holder:

            error = error_holder["error"]

            yield {
                "stage": "warning",
                "pct": 20,
                "log": (f"⚠️ Local brain failed, " f"using fallback: {error}"),
                "gallery": [],
                "video": None,
                "model_status": {},
            }

            bible = production_bible.generate_production_bible(
                story, use_local_brain=False
            )

            enhanced_story = story

        else:

            bible = result_holder["bible"]

            enhanced_story = result_holder.get("enhanced_script", story)

            # ---------------------------------------------
            # SAVE TO DRIVE
            # ---------------------------------------------

            cache_manager.save_brain_result(
                story=story, bible=bible, enhanced_script=enhanced_story
            )

            yield {
                "stage": "cache",
                "pct": 20,
                "log": (
                    "💾 BRAIN CACHE SAVED\n"
                    "   ├─ Production Bible → Google Drive\n"
                    "   └─ Enhanced script → Google Drive"
                ),
                "gallery": [],
                "video": None,
                "model_status": {},
            }

    # ---- Stage 4: Scene Orchestration ----
    yield {
        "stage": "scenes",
        "pct": 20,
        "log": f"📋 Breaking into {len(bible.scenes)} scenes...",
        "gallery": [],
        "video": None,
        "model_status": {},
    }

    orchestrator = scene_orchestrator.SceneOrchestrator(bible)
    production_units = orchestrator.breakdown()
    total_scenes = len(production_units)
    consistency = consistency_manager.ConsistencyManager(bible)

    # ---- Stage 4b: Auto-select the best video model for this job ----
    worker_id = worker_client.connected_worker_ids()[0] if has_workers else None
    chosen_model_id = (
        model_selector.select_video_model(
            production_units, worker_id, quality_preference
        )
        if model_id in (None, "", "auto")
        else model_id
    )
    model_status = model_selector.describe_selection(chosen_model_id, worker_id)

    yield {
        "stage": "model_select",
        "pct": 25,
        "log": f"🧠 Selected model: {model_status['selected_model_name']} "
        f"({'worker ' + worker_id if worker_id else 'local fallback'})",
        "gallery": [],
        "video": None,
        "model_status": model_status,
    }

    if worker_id:
        # Only one heavy video model stays resident on a worker at a time --
        # delete whatever else is installed there before asking it to
        # install the one we just picked.
        yield {
            "stage": "model_select",
            "pct": 27,
            "log": f"🧹 Making sure only {model_status['selected_model_name']} is installed on the worker...",
            "gallery": [],
            "video": None,
            "model_status": model_status,
        }
        worker_client.switch_video_model(worker_id, chosen_model_id)

    # ---- Stage 5: Generate Videos on Workers ----
    generated_clips = []

    if total_scenes == 0:
      yield {
        "stage": "error",
        "pct": 30,
        "log": "❌ No production scenes were created.",
        "gallery": [],
        "video": None,
        "model_status": model_status,
    }
      return


    for idx, unit in enumerate(production_units):

        pct = 30 + (idx / total_scenes) * 65

        # ========================================================
        # CHECK SCENE CACHE
        # ========================================================

        cached_video = cache_manager.load_scene(story, unit.scene_id)

        if cached_video:

            clip_entry = {
                "scene_id": unit.scene_id,
                "description": unit.description,
                "video_data": cached_video,
                "duration": duration_per_scene,
            }

            generated_clips.append(clip_entry)

            orchestrator.update_unit_status(unit.scene_id, "completed")

            yield {
                "stage": "cache",
                "pct": pct,
                "log": (
                    f"⚡ Scene {idx + 1}/{total_scenes}: "
                    f"Loaded from Drive cache\n"
                    f"   └─ Skipping video generation"
                ),
                "gallery": generated_clips,
                "video": None,
                "model_status": model_status,
            }

            continue

        # ========================================================
        # NO CACHE → GENERATE
        # ========================================================

        yield {
            "stage": "generating",
            "pct": pct,
            "log": (f"🎬 Scene {idx + 1}/{total_scenes}: " f"{unit.description[:50]}..."),
            "gallery": generated_clips,
            "video": None,
            "model_status": model_status,
        }

        try:

            # ====================================================
            # GENERATE ON WORKER
            # ====================================================

            if has_workers:

                yield {
                    "stage": "generating",
                    "pct": pct,
                    "log": (
                        f"🎬 Scene {idx + 1}/{total_scenes}: "
                        f"Sending to worker for generation..."
                    ),
                    "gallery": generated_clips,
                    "video": None,
                    "model_status": model_status,
                }

                result = worker_client.generate_video_on_worker(
                    worker_id=worker_id,
                    prompt=unit.visual_prompt,
                    model_id=chosen_model_id,
                    context=consistency.get_worker_context(unit),
                    duration_seconds=duration_per_scene,
                    fps=fps,
                    width=width,
                    height=height,
                    seed=unit.seed,
                )

            # ====================================================
            # LOCAL FALLBACK
            # ====================================================

            else:

                yield {
                    "stage": "generating",
                    "pct": pct,
                    "log": (
                        f"🎬 Scene {idx + 1}/{total_scenes}: "
                        f"No workers, using local fallback..."
                    ),
                    "gallery": generated_clips,
                    "video": None,
                    "model_status": model_status,
                }

                result = video_models.generate_fallback(
                    prompt=unit.visual_prompt,
                    duration=duration_per_scene,
                    fps=fps,
                    width=width,
                    height=height,
                    seed=unit.seed,
                )

            # ====================================================
            # ADD GENERATED CLIP
            # ====================================================

            video_data = result.get("video_data", "")

            clip_entry = {
                "scene_id": unit.scene_id,
                "description": unit.description,
                "video_data": video_data,
                "duration": result.get("duration", duration_per_scene),
            }

            generated_clips.append(clip_entry)

            orchestrator.update_unit_status(unit.scene_id, "completed")

            # ====================================================
            # SAVE COMPLETED CLIP TO DRIVE
            # ====================================================

            if video_data:

                saved_path = cache_manager.save_scene(
                    story=story, scene_id=unit.scene_id, video_data=video_data
                )

                if saved_path:

                    yield {
                        "stage": "cache",
                        "pct": pct,
                        "log": (
                            f"💾 Scene {idx + 1}/{total_scenes} "
                            f"saved to Drive cache\n"
                            f"   └─ {unit.scene_id}"
                        ),
                        "gallery": generated_clips,
                        "video": None,
                        "model_status": model_status,
                    }

        except Exception as e:

            yield {
                "stage": "error",
                "pct": pct,
                "log": (f"❌ Scene {idx + 1} failed: {e}"),
                "gallery": generated_clips,
                "video": None,
                "model_status": model_status,
            }

            continue

    # ---- Stage 6: Complete ----
    final_video = generated_clips[0]["video_data"] if generated_clips else None

    yield {
        "stage": "finished",
        "pct": 100,
        "log": f"✅ Complete! Generated {len(generated_clips)} clips.",
        "gallery": generated_clips,
        "video": final_video,
        "model_status": model_status,
    }
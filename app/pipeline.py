"""
Video Pipeline - Models download on workers, not main
"""

from typing import Generator, List, Dict, Any, Optional
import time

from app import (
    production_bible,
    scene_orchestrator,
    consistency_manager,
    worker_client,
    video_models,
    model_manager,
    model_selector,
    scene_analyzer,
    script_enhancer
)


def run_video_pipeline(
    story: str,
    model_id: str = "auto",
    duration_per_scene: int = 4,
    auto_download: bool = True,
    fps: int = 24,
    width: int = 576,
    height: int = 320,
    quality_preference: str = "auto"
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
            'stage': 'error',
            'pct': 0,
            'log': '⚠️ Please enter a longer story (minimum 20 characters)',
            'gallery': [],
            'video': None,
            'model_status': {}
        }
        return
    
    # ---- Stage 2: Check Workers ----
    has_workers = worker_client.is_any_connected()
    
    if not has_workers:
        yield {
            'stage': 'warning',
            'pct': 5,
            'log': '⚠️ No GPU workers connected. Video generation may be slow or unavailable.',
            'gallery': [],
            'video': None,
            'model_status': {}
        }
    
    # ---- Stage 3: Generate Production Bible ----
    yield {
        'stage': 'bible',
        'pct': 10,
        'log': '📖 Creating production bible...',
        'gallery': [],
        'video': None,
        'model_status': {}
    }
    
    try:
        bible = production_bible.generate_production_bible(story)
        enhanced_story = script_enhancer.enhance_script(story, bible)

        # The enhanced script used to be computed and then thrown away --
        # bible.scenes stayed based on the original, unenhanced story. If
        # enhancement actually changed the text, re-run scene analysis on
        # the enhanced version and re-enrich with the same bible context, so
        # scene descriptions/dialogue actually reflect the improved script.
        if enhanced_story and enhanced_story.strip() != story.strip():
            bible.scenes = production_bible._enrich_scenes_with_bible(
                scene_analyzer.analyze_story(enhanced_story), bible
            )
    except Exception as e:
        yield {
            'stage': 'error',
            'pct': 10,
            'log': f'❌ Bible generation failed: {e}',
            'gallery': [],
            'video': None,
            'model_status': {}
        }
        return
    
    # ---- Stage 4: Scene Orchestration ----
    yield {
        'stage': 'scenes',
        'pct': 20,
        'log': f'📋 Breaking into {len(bible.scenes)} scenes...',
        'gallery': [],
        'video': None,
        'model_status': {}
    }
    
    orchestrator = scene_orchestrator.SceneOrchestrator(bible)
    production_units = orchestrator.breakdown()
    total_scenes = len(production_units)
    consistency = consistency_manager.ConsistencyManager(bible)

    # ---- Stage 4b: Auto-select the best video model for this job ----
    worker_id = worker_client.connected_worker_ids()[0] if has_workers else None
    chosen_model_id = (
        model_selector.select_video_model(production_units, worker_id, quality_preference)
        if model_id in (None, "", "auto")
        else model_id
    )
    model_status = model_selector.describe_selection(chosen_model_id, worker_id)

    yield {
        'stage': 'model_select',
        'pct': 25,
        'log': f"🧠 Selected model: {model_status['selected_model_name']} "
               f"({'worker ' + worker_id if worker_id else 'local fallback'})",
        'gallery': [],
        'video': None,
        'model_status': model_status
    }

    if worker_id:
        # Only one heavy video model stays resident on a worker at a time --
        # delete whatever else is installed there before asking it to
        # install the one we just picked.
        yield {
            'stage': 'model_select',
            'pct': 27,
            'log': f"🧹 Making sure only {model_status['selected_model_name']} is installed on the worker...",
            'gallery': [],
            'video': None,
            'model_status': model_status
        }
        worker_client.switch_video_model(worker_id, chosen_model_id)

    # ---- Stage 5: Generate Videos on Workers ----
    generated_clips = []
    
    for idx, unit in enumerate(production_units):
        pct = 30 + (idx / total_scenes) * 65
        
        yield {
            'stage': 'generating',
            'pct': pct,
            'log': f'🎬 Scene {idx + 1}/{total_scenes}: {unit.description[:50]}...',
            'gallery': generated_clips,
            'video': None,
            'model_status': model_status
        }
        
        try:
            # ---- GENERATE ON WORKER ----
            if has_workers:
                # Send to worker - worker handles model download. Also send
                # the bible-derived consistency context (characters,
                # locations, style, seeds) instead of an empty dict, so
                # workers can actually keep scenes visually consistent.
                yield {
                    'stage': 'generating',
                    'pct': pct,
                    'log': f'🎬 Scene {idx + 1}/{total_scenes}: Sending to worker for generation...',
                    'gallery': generated_clips,
                    'video': None,
                    'model_status': model_status
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
                    seed=unit.seed
                )
            else:
                # Fallback: generate locally (will be slow, and never a
                # heavy diffusion model -- the main Colab only ever runs the
                # lightweight placeholder generator itself).
                yield {
                    'stage': 'generating',
                    'pct': pct,
                    'log': f'🎬 Scene {idx + 1}/{total_scenes}: No workers, using local fallback...',
                    'gallery': generated_clips,
                    'video': None,
                    'model_status': model_status
                }
                
                # Use local video generation (if available)
                result = video_models.generate_fallback(
                    prompt=unit.visual_prompt,
                    duration=duration_per_scene,
                    fps=fps,
                    width=width,
                    height=height,
                    seed=unit.seed
                )
            
            # Add to gallery
            clip_entry = {
                'scene_id': unit.scene_id,
                'description': unit.description,
                'video_data': result.get('video_data', ''),
                'duration': result.get('duration', duration_per_scene)
            }
            generated_clips.append(clip_entry)
            
            # Update status
            orchestrator.update_unit_status(unit.scene_id, 'completed')
            
        except Exception as e:
            yield {
                'stage': 'error',
                'pct': pct,
                'log': f'❌ Scene {idx + 1} failed: {e}',
                'gallery': generated_clips,
                'video': None,
                'model_status': model_status
            }
            continue
    
    # ---- Stage 6: Complete ----
    final_video = generated_clips[0]['video_data'] if generated_clips else None
    
    yield {
        'stage': 'finished',
        'pct': 100,
        'log': f'✅ Complete! Generated {len(generated_clips)} clips.',
        'gallery': generated_clips,
        'video': final_video,
        'model_status': model_status
    }
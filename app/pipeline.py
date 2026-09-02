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
    scene_analyzer,
    script_enhancer
)


def run_video_pipeline(
    story: str,
    model_id: str,
    duration_per_scene: int = 4,
    auto_download: bool = True,
    fps: int = 24,
    width: int = 576,
    height: int = 320
) -> Generator[Dict[str, Any], None, None]:
    """
    Run complete video pipeline.
    Models download on workers, not main.
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
            'model_status': {}
        }
        
        try:
            # ---- GENERATE ON WORKER ----
            if has_workers:
                # Send to worker - worker handles model download
                yield {
                    'stage': 'generating',
                    'pct': pct,
                    'log': f'🎬 Scene {idx + 1}/{total_scenes}: Sending to worker for generation...',
                    'gallery': generated_clips,
                    'video': None,
                    'model_status': {}
                }
                
                result = worker_client.generate_video_on_worker(
                    worker_id=worker_client.connected_worker_ids()[0],  # First connected worker
                    prompt=unit.visual_prompt,
                    model_id=model_id,
                    context={},
                    duration_seconds=duration_per_scene,
                    fps=fps,
                    width=width,
                    height=height,
                    seed=unit.seed
                )
            else:
                # Fallback: generate locally (will be slow)
                yield {
                    'stage': 'generating',
                    'pct': pct,
                    'log': f'🎬 Scene {idx + 1}/{total_scenes}: No workers, using local fallback...',
                    'gallery': generated_clips,
                    'video': None,
                    'model_status': {}
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
                'model_status': {}
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
        'model_status': {}
    }
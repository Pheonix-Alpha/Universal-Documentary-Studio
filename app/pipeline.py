"""
Video Pipeline - Complete with auto-download
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
    Run complete video pipeline with auto-download.
    Yields progress updates.
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
    
    # ---- Stage 2: Generate Production Bible ----
    yield {
        'stage': 'bible',
        'pct': 5,
        'log': '📖 Creating production bible...',
        'gallery': [],
        'video': None,
        'model_status': model_manager.get_vram_status()
    }
    
    try:
        bible = production_bible.generate_production_bible(story)
        enhanced_story = script_enhancer.enhance_script(story, bible)
    except Exception as e:
        yield {
            'stage': 'error',
            'pct': 5,
            'log': f'❌ Bible generation failed: {e}',
            'gallery': [],
            'video': None,
            'model_status': {}
        }
        return
    
    # ---- Stage 3: Scene Orchestration ----
    yield {
        'stage': 'scenes',
        'pct': 15,
        'log': f'📋 Breaking into {len(bible.scenes)} scenes...',
        'gallery': [],
        'video': None,
        'model_status': model_manager.get_vram_status()
    }
    
    orchestrator = scene_orchestrator.SceneOrchestrator(bible)
    production_units = orchestrator.breakdown()
    total_scenes = len(production_units)
    
    # ---- Stage 4: Auto-Download Model ----
    if auto_download:
        yield {
            'stage': 'model',
            'pct': 20,
            'log': f'📥 Checking/downloading model: {model_id}...',
            'gallery': [],
            'video': None,
            'model_status': model_manager.get_vram_status()
        }
        
        try:
            # Auto-download and load model
            def progress_callback(pct, msg):
                print(f"  [{pct}%] {msg}")
            
            model_manager.auto_download_and_load_model(
                model_id,
                progress_callback=progress_callback
            )
        except Exception as e:
            yield {
                'stage': 'error',
                'pct': 20,
                'log': f'❌ Model download failed: {e}',
                'gallery': [],
                'video': None,
                'model_status': {}
            }
            return
    
    # ---- Stage 5: Generate Videos ----
    generated_clips = []
    
    for idx, unit in enumerate(production_units):
        pct = 25 + (idx / total_scenes) * 70
        
        yield {
            'stage': 'generating',
            'pct': pct,
            'log': f'🎬 Scene {idx + 1}/{total_scenes}: {unit.description[:50]}...',
            'gallery': generated_clips,
            'video': None,
            'model_status': model_manager.get_vram_status()
        }
        
        try:
            # Generate video
            result = video_models.generate_video(
                prompt=unit.visual_prompt,
                model_id=model_id,
                context={},
                duration_seconds=duration_per_scene,
                fps=fps,
                width=width,
                height=height,
                seed=unit.seed
            )
            
            # Add to gallery
            clip_entry = {
                'scene_id': unit.scene_id,
                'description': unit.description,
                'video_data': result['video_data'],
                'duration': result['duration']
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
                'model_status': model_manager.get_vram_status()
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
        'model_status': model_manager.get_vram_status()
    }
    
    # ---- Stage 7: Auto-Cleanup ----
    # Model stays loaded for potential reuse, but can be unloaded manually
    # or will be auto-unloaded when space is needed
    print("✅ Pipeline complete. Model kept in VRAM for reuse.")
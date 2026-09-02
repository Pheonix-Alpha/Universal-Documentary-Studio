"""
Video Pipeline - The main flow for video generation
"""

from typing import Generator, List, Dict, Any, Optional
import json
import os
import time

from app import (
    config,
    compute,
    scene_analyzer,
    script_enhancer,
    production_bible,
    scene_orchestrator,
    consistency_manager,
    worker_client,
    video_models
)


def run_video_pipeline(
    story: str,
    clip_model_id: str,
    video_model_id: str,
    top_k: int = 3,
    use_duckduckgo: bool = True,
    duration_per_scene: int = 4,
    fps: int = 24,
    width: int = 576,
    height: int = 320
) -> Generator[Dict[str, Any], None, None]:
    """
    Run the complete video generation pipeline.
    
    Yields:
        {
            'stage': str,  # 'bible', 'scenes', 'generating', 'assembling', 'finished'
            'pct': float,
            'log': str,
            'gallery': List[Dict],
            'video': Optional[str],  # base64 encoded final video
            'bible': Optional[Dict]  # The production bible
        }
    """
    
    # ---- Stage 1: Validate ----
    if not story or len(story.strip()) < 10:
        yield {
            'stage': 'error',
            'pct': 0,
            'log': 'Please enter a longer story.',
            'gallery': [],
            'video': None,
            'bible': None
        }
        return
    
    # ---- Stage 2: Generate Production Bible ----
    yield {
        'stage': 'bible',
        'pct': 5,
        'log': '🎬 Creating production bible...',
        'gallery': [],
        'video': None,
        'bible': None
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
            'bible': None
        }
        return
    
    # ---- Stage 3: Scene Orchestration ----
    yield {
        'stage': 'scenes',
        'pct': 15,
        'log': f'📋 Breaking into {len(bible.scenes)} scenes...',
        'gallery': [],
        'video': None,
        'bible': bible.to_dict()
    }
    
    orchestrator = scene_orchestrator.SceneOrchestrator(bible)
    production_units = orchestrator.breakdown()
    
    consistency = consistency_manager.ConsistencyManager(bible)
    
    # ---- Stage 4: Generate Reference Images (for character consistency) ----
    yield {
        'stage': 'references',
        'pct': 20,
        'log': '🖼️ Generating character reference images...',
        'gallery': [],
        'video': None,
        'bible': bible.to_dict()
    }
    
    # TODO: Generate reference images for each character
    
    # ---- Stage 5: Distribute Video Generation ----
    total_scenes = len(production_units)
    generated_clips = []
    
    # Check if workers are available
    has_workers = worker_client.is_any_connected()
    has_local_video = video_models.is_video_model_installed(video_model_id)
    
    if not has_workers and not has_local_video:
        yield {
            'stage': 'error',
            'pct': 25,
            'log': '❌ No video generation capable. Connect a worker or install a local video model.',
            'gallery': [],
            'video': None,
            'bible': bible.to_dict()
        }
        return
    
    for idx, unit in enumerate(production_units):
        pct = 25 + (idx / total_scenes) * 60
        
        yield {
            'stage': 'generating',
            'pct': pct,
            'log': f'🎬 Generating scene {unit.scene_id + 1}/{total_scenes}: {unit.description[:50]}...',
            'gallery': generated_clips,
            'video': None,
            'bible': bible.to_dict()
        }
        
        try:
            # Get consistency context
            context = consistency.get_worker_context(unit)
            
            # Generate video (use workers if available, else local)
            if has_workers:
                result = worker_client.generate_video_round_robin(
                    prompt=unit.visual_prompt,
                    model_id=video_model_id,
                    context=context,
                    duration_seconds=duration_per_scene,
                    fps=fps,
                    width=width,
                    height=height,
                    seed=unit.seed
                )
            else:
                result = video_models.generate_video(
                    prompt=unit.visual_prompt,
                    model_id=video_model_id,
                    context=context,
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
                'duration': result['duration'],
                'metadata': result['metadata'],
                'worker': unit.assigned_worker
            }
            generated_clips.append(clip_entry)
            
            # Update unit status
            orchestrator.update_unit_status(unit.scene_id, 'completed')
            
        except Exception as e:
            yield {
                'stage': 'error',
                'pct': pct,
                'log': f'❌ Scene {unit.scene_id + 1} failed: {e}',
                'gallery': generated_clips,
                'video': None,
                'bible': bible.to_dict()
            }
            orchestrator.update_unit_status(unit.scene_id, 'failed')
            continue
    
    # ---- Stage 6: Assemble Final Video ----
    if generated_clips:
        yield {
            'stage': 'assembling',
            'pct': 90,
            'log': f'✂️ Assembling {len(generated_clips)} clips into final video...',
            'gallery': generated_clips,
            'video': None,
            'bible': bible.to_dict()
        }
        
        # TODO: Video assembly
        # Use ffmpeg or moviepy to concatenate clips
        final_video = _assemble_video(generated_clips)
    else:
        final_video = None
    
    # ---- Stage 7: Complete ----
    yield {
        'stage': 'finished',
        'pct': 100,
        'log': f'✅ Complete! Generated {len(generated_clips)} clips.',
        'gallery': generated_clips,
        'video': final_video,
        'bible': bible.to_dict()
    }


def _assemble_video(clips: List[Dict[str, Any]]) -> Optional[str]:
    """Assemble clips into a final video"""
    # Placeholder - would use ffmpeg or moviepy
    if clips and clips[0].get('video_data'):
        return clips[0]['video_data']  # Return first clip for now
    return None
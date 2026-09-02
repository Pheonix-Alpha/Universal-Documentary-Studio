"""
Video Models - Enhanced with better generation
"""

import os
import base64
import tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import cv2
from typing import List, Dict, Any, Optional


VIDEO_MODEL_REGISTRY = {
    'cerspense/zeroscope_v2_576w': {
        'name': 'ZeroScope v2',
        'type': 'text2vid',
        'size_gb': 7.2,
        'description': 'Text-to-video - good for general use'
    },
    'stabilityai/stable-video-diffusion-img2vid': {
        'name': 'Stable Video Diffusion',
        'type': 'img2vid',
        'size_gb': 9.5,
        'description': 'Image-to-video - best quality'
    }
}


def generate_video(
    prompt: str,
    model_id: str,
    context: Dict[str, Any] = None,
    duration_seconds: int = 4,
    fps: int = 24,
    width: int = 576,
    height: int = 320,
    seed: int = 42,
    reference_image: Optional[str] = None,
    callback=None
) -> Dict[str, Any]:
    """Generate video using loaded model"""
    
    print(f"🎬 Generating video with model: {model_id}")
    print(f"   Prompt: {prompt[:100]}...")
    
    try:
        # Check if model is in registry
        if model_id in VIDEO_MODEL_REGISTRY:
            # Try to use the loaded model from model_manager
            from app import model_manager
            
            loaded = model_manager._loaded_models.get(model_id)
            if loaded and loaded.get('pipeline'):
                return _generate_with_pipeline(
                    loaded['pipeline'],
                    prompt,
                    duration_seconds,
                    fps,
                    width,
                    height,
                    seed,
                    reference_image
                )
        
        # Try to generate with diffusers directly
        return _generate_with_diffusers(
            model_id,
            prompt,
            duration_seconds,
            fps,
            width,
            height,
            seed
        )
        
    except Exception as e:
        print(f"⚠️ Real model generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Fallback: generate a simple video
    print("🔄 Using fallback video generation")
    return _generate_fallback_video(
        prompt,
        duration_seconds,
        fps,
        width,
        height,
        seed
    )


def _generate_with_pipeline(
    pipeline,
    prompt: str,
    duration: int,
    fps: int,
    width: int,
    height: int,
    seed: int,
    reference_image: Optional[str]
) -> Dict[str, Any]:
    """Generate using loaded pipeline"""
    import torch
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # Handle different pipeline types
        if hasattr(pipeline, 'generate'):
            # Some pipelines use generate()
            frames = pipeline.generate(
                prompt=prompt,
                num_frames=min(duration * fps, 24),
                height=height,
                width=width,
                generator=generator
            )
        elif hasattr(pipeline, '__call__'):
            # Standard diffusion pipeline
            frames = pipeline(
                prompt,
                num_frames=min(duration * fps, 24),
                num_inference_steps=20,
                height=height,
                width=width,
                generator=generator
            ).frames
        
        if frames:
            video_bytes = _frames_to_video_bytes(frames, fps)
            return {
                'video_data': base64.b64encode(video_bytes).decode('utf-8'),
                'duration': duration,
                'fps': fps,
                'width': width,
                'height': height,
                'metadata': {'seed': seed, 'model': 'pipeline'}
            }
        
    except Exception as e:
        print(f"Pipeline generation failed: {e}")
        raise
    
    raise RuntimeError("Pipeline generation returned no frames")


def _generate_with_diffusers(
    model_id: str,
    prompt: str,
    duration: int,
    fps: int,
    width: int,
    height: int,
    seed: int
) -> Dict[str, Any]:
    """Generate using diffusers directly"""
    try:
        from diffusers import DiffusionPipeline
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float16 if device == 'cuda' else torch.float32
        
        # Use zeroscope if available
        if 'zeroscope' in model_id.lower():
            model_path = "cerspense/zeroscope_v2_576w"
        else:
            model_path = model_id
        
        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            dtype=dtype,
            low_cpu_mem_usage=True
        )
        
        # Don't call eval() if not available
        if hasattr(pipe, 'eval'):
            pipe.eval()
        
        # Move to device
        if device == 'cuda':
            try:
                pipe.enable_model_cpu_offload()
            except:
                pipe.to(device)
        else:
            pipe.to('cpu')
        
        generator = torch.Generator(device='cpu').manual_seed(seed)
        
        frames = pipe(
            prompt,
            num_frames=min(duration * fps, 24),
            num_inference_steps=20,
            height=height,
            width=width,
            generator=generator
        ).frames
        
        video_bytes = _frames_to_video_bytes(frames, fps)
        
        return {
            'video_data': base64.b64encode(video_bytes).decode('utf-8'),
            'duration': duration,
            'fps': fps,
            'width': width,
            'height': height,
            'metadata': {'seed': seed, 'model': model_id}
        }
        
    except Exception as e:
        raise RuntimeError(f"Diffusers generation failed: {e}")


def _generate_fallback_video(
    prompt: str,
    duration: int,
    fps: int,
    width: int,
    height: int,
    seed: int
) -> Dict[str, Any]:
    """Generate a simple animated fallback video - BETTER VERSION"""
    try:
        import numpy as np
        
        num_frames = min(duration * fps, 30)
        frames = []
        
        # Use seed
        np.random.seed(seed)
        
        for i in range(num_frames):
            # Create frame with gradient
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Animated gradient
            phase = i / max(1, num_frames)
            r = int(80 + 155 * (0.5 + 0.5 * np.sin(phase * 2 * np.pi + seed)))
            g = int(80 + 155 * (0.5 + 0.5 * np.sin(phase * 2 * np.pi + 2.094 + seed)))
            b = int(80 + 155 * (0.5 + 0.5 * np.sin(phase * 2 * np.pi + 4.188 + seed)))
            
            for y in range(height):
                ratio = y / max(1, height)
                color = (
                    int(r * (0.5 + 0.5 * (1 - ratio))),
                    int(g * (0.5 + 0.5 * (1 - ratio))),
                    int(b * (0.5 + 0.5 * (1 - ratio)))
                )
                frame[y, :] = color
            
            # Add text with better formatting
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Split prompt into lines
            text_lines = prompt.split('.')[:3]
            lines = []
            for line in text_lines:
                if len(line) > 30:
                    words = line.split()
                    current_line = []
                    for word in words:
                        if len(' '.join(current_line + [word])) <= 30:
                            current_line.append(word)
                        else:
                            lines.append(' '.join(current_line))
                            current_line = [word]
                    if current_line:
                        lines.append(' '.join(current_line))
                else:
                    lines.append(line.strip())
            
            # Draw text
            for idx, line in enumerate(lines):
                y_pos = height // 2 - (len(lines) * 30) // 2 + idx * 30
                cv2.putText(frame, line, (10, y_pos), font, 0.5, (255, 255, 255), 1)
            
            # Add "Generating..." indicator
            if i < num_frames - 1:
                progress = (i / num_frames) * 100
                cv2.putText(frame, f"Generating: {int(progress)}%", (10, height - 20), font, 0.4, (200, 200, 200), 1)
            else:
                cv2.putText(frame, "Complete!", (10, height - 20), font, 0.4, (0, 255, 0), 1)
            
            frames.append(frame)
        
        video_bytes = _frames_to_video_bytes(frames, fps)
        
        return {
            'video_data': base64.b64encode(video_bytes).decode('utf-8'),
            'duration': duration,
            'fps': fps,
            'width': width,
            'height': height,
            'metadata': {
                'model': 'fallback',
                'seed': seed,
                'note': 'Real model failed - using fallback'
            }
        }
        
    except Exception as e:
        print(f"Fallback generation failed: {e}")
        return {
            'video_data': '',
            'duration': duration,
            'fps': fps,
            'width': width,
            'height': height,
            'metadata': {'error': str(e)}
        }


def _frames_to_video_bytes(frames: List[np.ndarray], fps: int) -> bytes:
    """Convert frames to video bytes"""
    if not frames:
        return b''
    
    try:
        import cv2
        
        height, width = frames[0].shape[:2]
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(tmp.name, fourcc, fps, (width, height))
            
            for frame in frames:
                if isinstance(frame, np.ndarray):
                    if len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    out.write(frame)
            
            out.release()
            
            with open(tmp.name, 'rb') as f:
                video_bytes = f.read()
            
            os.unlink(tmp.name)
            return video_bytes
            
    except Exception as e:
        print(f"Failed to convert frames: {e}")
        return b''
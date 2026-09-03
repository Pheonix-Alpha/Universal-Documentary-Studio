"""
Video Models - Complete implementation with fallback handling
"""

from typing import List, Dict, Any, Optional
import os
import torch
import base64
import tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import traceback

# Video model registry
VIDEO_MODEL_REGISTRY = {
    'stabilityai/stable-video-diffusion-img2vid': {
        'name': 'Stable Video Diffusion',
        'type': 'img2vid',
        'size_gb': 9.5,
        'description': 'Image-to-video generation - best quality'
    },
    'cerspense/zeroscope_v2_576w': {
        'name': 'ZeroScope v2',
        'type': 'text2vid',
        'size_gb': 7.2,
        'description': 'Text-to-video - good for general use'
    }
}


def list_video_models() -> List[Dict[str, Any]]:
    """List available video models"""
    models = []
    for model_id, info in VIDEO_MODEL_REGISTRY.items():
        models.append({
            'id': model_id,
            'name': info['name'],
            'type': info['type'],
            'size_gb': info.get('size_gb', 0),
            'description': info['description'],
            'installed': is_video_model_installed(model_id)
        })
    return models


def is_video_model_installed(model_id: str) -> bool:
    """Check if a video model is installed"""
    try:
        from app import model_manager
        return model_manager.is_installed(model_id)
    except:
        return False


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
    """
    Generate a video clip using the specified model.
    Returns base64-encoded video data.
    """
    print(f"🎬 Generating video with model: {model_id}")
    print(f"   Prompt: {prompt[:100]}...")
    
    # Try to use the real model
    try:
        # Check if we should use a real model
        if model_id in VIDEO_MODEL_REGISTRY:
            model_info = VIDEO_MODEL_REGISTRY[model_id]
            
            if model_info['type'] == 'img2vid':
                result = _generate_img2vid(
                    prompt, model_id, context, duration_seconds, fps,
                    width, height, seed, reference_image, callback
                )
            elif model_info['type'] == 'text2vid':
                result = _generate_text2vid(
                    prompt, model_id, context, duration_seconds, fps,
                    width, height, seed, callback
                )
            else:
                result = _generate_fallback_video(
                    prompt, duration_seconds, fps, width, height, seed
                )
            
            # Check if generation succeeded
            if result and result.get('video_data'):
                print(f"✅ Video generated: {len(result['video_data'])} bytes")
                return result
    
    except Exception as e:
        print(f"⚠️ Real model generation failed: {e}")
        print(traceback.format_exc())
    
    # Fallback: generate a simple animated video
    print("🔄 Using fallback video generation")
    return _generate_fallback_video(prompt, duration_seconds, fps, width, height, seed)


def _generate_img2vid(
    prompt: str,
    model_id: str,
    context: Dict[str, Any],
    duration: int,
    fps: int,
    width: int,
    height: int,
    seed: int,
    reference_image: Optional[str],
    callback=None
) -> Dict[str, Any]:
    """Generate video using Stable Video Diffusion"""
    try:
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image
        
        model_path = _get_model_path(model_id)
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise ValueError(f"Model not found at {model_path}")
        
        # Load pipeline
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float16 if device == 'cuda' else torch.float32
        
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        
        if device == 'cuda':
            pipe.enable_model_cpu_offload()
            pipe.enable_attention_slicing()
        
        # Load or create reference image
        if reference_image:
            import base64
            from io import BytesIO
            if reference_image.startswith('data:image'):
                img_data = reference_image.split(',')[1]
                img = Image.open(BytesIO(base64.b64decode(img_data)))
            else:
                img = load_image(reference_image)
        else:
            img = _create_placeholder_image(prompt, width, height)
        
        # Generate
        generator = torch.Generator(device=device).manual_seed(seed)
        frames = pipe(
            img,
            decode_chunk_size=2,
            generator=generator,
            num_frames=min(duration * fps, 24)
        ).frames[0]
        
        # Convert to video
        video_bytes = _frames_to_video_bytes(frames, fps)
        
        return {
            'video_data': base64.b64encode(video_bytes).decode('utf-8'),
            'duration': duration,
            'fps': fps,
            'width': width,
            'height': height,
            'metadata': {'model': model_id, 'seed': seed}
        }
        
    except Exception as e:
        print(f"❌ SVD generation failed: {e}")
        raise


def _generate_text2vid(
    prompt: str,
    model_id: str,
    context: Dict[str, Any],
    duration: int,
    fps: int,
    width: int,
    height: int,
    seed: int,
    callback=None
) -> Dict[str, Any]:
    """Generate video using text-to-video models"""
    try:
        from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float16 if device == 'cuda' else torch.float32
        
        # Use ZeroScope if available
        if 'zeroscope' in model_id.lower():
            model_path = 'cerspense/zeroscope_v2_576w'
        else:
            model_path = _get_model_path(model_id)
        
        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        if device == 'cuda':
            pipe.enable_model_cpu_offload()
        
        # Generate
        generator = torch.Generator(device=device).manual_seed(seed)
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
            'metadata': {'model': model_id, 'seed': seed}
        }
        
    except Exception as e:
        print(f"❌ Text2Video generation failed: {e}")
        raise


def generate_fallback(
    prompt: str,
    duration: int = 4,
    fps: int = 24,
    width: int = 576,
    height: int = 320,
    seed: int = 42
) -> Dict[str, Any]:
    """Public entry point for a keyword-argument call (prompt=..., duration=...)
    to the local, no-GPU-needed placeholder video generator. pipeline.py's
    "no workers connected" branch called this name directly; it didn't
    exist before (only the private, positional-only `_generate_fallback_video`
    did), so that branch raised AttributeError every time it ran -- i.e.
    every time you tried to generate without a GPU worker connected."""
    return _generate_fallback_video(prompt, duration, fps, width, height, seed)


def _generate_fallback_video(
    prompt: str,
    duration: int,
    fps: int,
    width: int,
    height: int,
    seed: int
) -> Dict[str, Any]:
    """Generate a simple animated fallback video"""
    try:
        import cv2
        import numpy as np
        
        num_frames = min(duration * fps, 30)
        frames = []
        
        # Use seed for reproducibility
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
            
            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = prompt[:50]
            text_size = cv2.getTextSize(text, font, 0.6, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2
            
            # Add text shadow
            cv2.putText(frame, text, (text_x + 1, text_y + 1), font, 0.6, (0, 0, 0), 2)
            cv2.putText(frame, text, (text_x, text_y), font, 0.6, (255, 255, 255), 2)
            
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
                'note': 'Fallback video generated - install real models for better quality'
            }
        }
        
    except Exception as e:
        print(f"❌ Fallback generation failed: {e}")
        # Return empty video data
        return {
            'video_data': '',
            'duration': duration,
            'fps': fps,
            'width': width,
            'height': height,
            'metadata': {'error': str(e)}
        }


def _get_model_path(model_id: str) -> str:
    """Get model path from model_manager"""
    try:
        from app import model_manager
        return model_manager.get_model_path(model_id)
    except:
        # Fallback path
        return os.path.join(os.path.expanduser('~'), '.cache', 'models', model_id.replace('/', '__'))


def _create_placeholder_image(prompt: str, width: int, height: int) -> Image.Image:
    """Create a placeholder image with text"""
    img = Image.new('RGB', (width, height), color='navy')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    text = prompt[:100]
    draw.text((10, height//2 - 10), text, fill='white', font=font)
    
    return img


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
        print(f"❌ Failed to convert frames to video: {e}")
        return b''
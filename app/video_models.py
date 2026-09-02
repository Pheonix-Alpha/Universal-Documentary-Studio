"""
Video Models - Interface for various video generation models
Supports: Stable Video Diffusion, AnimateDiff, ModelScope, ZeroScope
"""

from typing import List, Dict, Any, Optional, Generator
import os
import torch
from PIL import Image
import numpy as np
import subprocess
import tempfile
import base64
from io import BytesIO
import json
import time

from app import config
from app import model_manager


# Video model registry - MUST BE DEFINED
VIDEO_MODEL_REGISTRY = {
    'stabilityai/stable-video-diffusion-img2vid': {
        'name': 'Stable Video Diffusion (Image to Video)',
        'type': 'img2vid',
        'size_gb': 9.5,
        'description': 'Generates video from an image - best quality',
        'vram_gb': 12.0
    },
    'cerspense/zeroscope_v2_576w': {
        'name': 'ZeroScope v2 (Text to Video)',
        'type': 'text2vid',
        'size_gb': 7.2,
        'description': 'Open-source text to video - good for general use',
        'vram_gb': 8.0
    },
    'damo-vilab/modelscope-damo-text-to-video-synthesis': {
        'name': 'ModelScope Text2Video',
        'type': 'text2vid',
        'size_gb': 11.3,
        'description': 'Text to video synthesis - high quality',
        'vram_gb': 14.0
    },
    'SG161222/Realistic_Vision_V5.1_noVAE': {
        'name': 'Realistic Vision (AnimateDiff)',
        'type': 'animate_diff',
        'size_gb': 4.8,
        'description': 'AnimateDiff for realistic video - requires motion module',
        'vram_gb': 8.0
    }
}


def list_video_models() -> List[Dict[str, Any]]:
    """List all available video models with install status"""
    models = []
    for model_id, info in VIDEO_MODEL_REGISTRY.items():
        installed = model_manager.is_installed(model_id) if hasattr(model_manager, 'is_installed') else False
        models.append({
            'id': model_id,
            'name': info['name'],
            'type': info['type'],
            'size_gb': info.get('size_gb', 0),
            'description': info['description'],
            'installed': installed,
            'vram_gb': info.get('vram_gb', 0)
        })
    return models


def is_video_model_installed(model_id: str) -> bool:
    """Check if a video model is installed"""
    if not hasattr(model_manager, 'is_installed'):
        return False
    
    # Check if it's in the main registry first
    if model_manager.is_installed(model_id):
        return True
    
    # Check video-specific path
    model_path = _get_video_model_path(model_id)
    if not os.path.exists(model_path):
        return False
    
    # Check for model files
    required_files = ['model_index.json', 'pipeline.json', 'config.json']
    for f in required_files:
        if not os.path.exists(os.path.join(model_path, f)):
            return False
    
    return True


def _get_video_model_path(model_id: str) -> str:
    """Get the local path for a video model"""
    model_dir = os.path.join(config.MODEL_DIR, 'video_models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Convert HF-style model id to safe path
    safe_id = model_id.replace('/', '__')
    return os.path.join(model_dir, safe_id)


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
    callback=None  # Progress callback
) -> Dict[str, Any]:
    """
    Generate a video clip using the specified model.
    
    Returns:
        {
            'video_data': base64_encoded_video,
            'duration': int,
            'fps': int,
            'width': int,
            'height': int,
            'metadata': dict
        }
    """
    model_info = VIDEO_MODEL_REGISTRY.get(model_id)
    if not model_info:
        raise ValueError(f"Unknown video model: {model_id}")
    
    if not is_video_model_installed(model_id):
        raise ValueError(f"Video model not installed: {model_id}")
    
    model_type = model_info['type']
    
    # Dispatch to appropriate generator
    if model_type == 'img2vid':
        return _generate_img2vid(
            prompt, model_id, context, duration_seconds, fps,
            width, height, seed, reference_image, callback
        )
    elif model_type == 'animate_diff':
        return _generate_animate_diff(
            prompt, model_id, context, duration_seconds, fps,
            width, height, seed, callback
        )
    elif model_type == 'text2vid':
        return _generate_text2vid(
            prompt, model_id, context, duration_seconds, fps,
            width, height, seed, callback
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


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
        from diffusers.utils import load_image, export_to_video
        import torch
        
        model_path = _get_video_model_path(model_id)
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise ValueError(f"Model not found at {model_path}")
        
        # Load pipeline with memory optimization
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            variant="fp16" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        
        if torch.cuda.is_available():
            pipe.enable_model_cpu_offload()
            pipe.enable_attention_slicing()
        
        # Load reference image
        if reference_image:
            if reference_image.startswith('data:image'):
                img_data = reference_image.split(',')[1]
                img = Image.open(BytesIO(base64.b64decode(img_data)))
            else:
                img = load_image(reference_image)
        else:
            # Generate a placeholder image from prompt
            img = _generate_placeholder_image(prompt, width, height)
        
        # Generate video
        generator = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu').manual_seed(seed)
        frames = pipe(
            img,
            decode_chunk_size=2,  # Lower = less VRAM usage
            generator=generator,
            num_frames=min(duration * fps, 24)  # SVD is limited to ~24 frames
        ).frames[0]
        
        # Export to video bytes
        video_bytes = _frames_to_video_bytes(frames, fps)
        
        return {
            'video_data': base64.b64encode(video_bytes).decode('utf-8'),
            'duration': duration,
            'fps': fps,
            'width': width,
            'height': height,
            'metadata': {'model': model_id, 'seed': seed}
        }
        
    except ImportError as e:
        print(f"Import error: {e}")
        return _generate_fallback_video(prompt, duration, fps, width, height, seed)
    except Exception as e:
        print(f"Video generation failed: {e}")
        return _generate_fallback_video(prompt, duration, fps, width, height, seed)


def _generate_animate_diff(
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
    """Generate video using AnimateDiff"""
    try:
        # Try to import animate_diff
        from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
        import torch
        
        model_path = _get_video_model_path(model_id)
        
        # Load base model
        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        if torch.cuda.is_available():
            pipe.enable_model_cpu_offload()
        
        # Generate frames
        generator = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu').manual_seed(seed)
        video_frames = pipe(
            prompt,
            num_frames=min(duration * fps, 16),
            num_inference_steps=20,
            height=height,
            width=width,
            generator=generator
        ).frames
        
        # Export
        video_bytes = _frames_to_video_bytes(video_frames, fps)
        
        return {
            'video_data': base64.b64encode(video_bytes).decode('utf-8'),
            'duration': duration,
            'fps': fps,
            'width': width,
            'height': height,
            'metadata': {'model': model_id, 'seed': seed}
        }
        
    except Exception as e:
        print(f"AnimateDiff generation failed: {e}")
        return _generate_fallback_video(prompt, duration, fps, width, height, seed)


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
        # Check if it's ModelScope
        if 'modelscope' in model_id.lower():
            return _generate_modelscope(prompt, duration, fps, width, height, seed)
        else:
            return _generate_zeroscope(prompt, duration, fps, width, height, seed)
            
    except Exception as e:
        print(f"Text2Video generation failed: {e}")
        return _generate_fallback_video(prompt, duration, fps, width, height, seed)


def _generate_modelscope(
    prompt: str,
    duration: int,
    fps: int,
    width: int,
    height: int,
    seed: int
) -> Dict[str, Any]:
    """Generate using ModelScope"""
    try:
        from modelscope.pipelines import pipeline
        from modelscope.outputs import OutputKeys
        
        pipe = pipeline('text-to-video-synthesis', model='damo-vilab/modelscope-damo-text-to-video-synthesis')
        
        # Generate
        output = pipe({
            'text': prompt,
            'seed': seed
        })
        
        # Get video
        video_path = output[OutputKeys.OUTPUT_VIDEO]
        
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
        
        return {
            'video_data': base64.b64encode(video_bytes).decode('utf-8'),
            'duration': duration,
            'fps': fps,
            'width': width,
            'height': height,
            'metadata': {'model': 'modelscope', 'seed': seed}
        }
        
    except ImportError:
        raise RuntimeError("ModelScope not installed. Run: pip install modelscope")
    except Exception as e:
        raise RuntimeError(f"ModelScope generation failed: {e}")


def _generate_zeroscope(
    prompt: str,
    duration: int,
    fps: int,
    width: int,
    height: int,
    seed: int
) -> Dict[str, Any]:
    """Generate using ZeroScope"""
    try:
        from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
        import torch
        
        model_id = "cerspense/zeroscope_v2_576w"
        
        pipe = DiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        if torch.cuda.is_available():
            pipe.enable_model_cpu_offload()
        
        # Generate frames
        generator = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu').manual_seed(seed)
        video_frames = pipe(
            prompt,
            num_frames=min(duration * fps, 24),
            num_inference_steps=20,
            height=height,
            width=width,
            generator=generator
        ).frames
        
        # Export
        video_bytes = _frames_to_video_bytes(video_frames, fps)
        
        return {
            'video_data': base64.b64encode(video_bytes).decode('utf-8'),
            'duration': duration,
            'fps': fps,
            'width': width,
            'height': height,
            'metadata': {'model': 'zeroscope', 'seed': seed}
        }
        
    except ImportError:
        raise RuntimeError("Diffusers not installed properly")
    except Exception as e:
        raise RuntimeError(f"ZeroScope generation failed: {e}")


def _generate_placeholder_image(prompt: str, width: int, height: int) -> Image.Image:
    """Generate a placeholder image from prompt using simple text overlay"""
    from PIL import Image, ImageDraw, ImageFont
    
    img = Image.new('RGB', (width, height), color='navy')
    draw = ImageDraw.Draw(img)
    
    # Try to use a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Draw text
    text = prompt[:50] + "..." if len(prompt) > 50 else prompt
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    draw.text((x, y), text, fill='white', font=font)
    
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
            
    except ImportError:
        # Fallback - return empty
        return b''


def _generate_fallback_video(
    prompt: str,
    duration: int,
    fps: int,
    width: int,
    height: int,
    seed: int
) -> Dict[str, Any]:
    """
    Fallback video generation - creates a simple animation with text overlay.
    Useful when no video models are available.
    """
    try:
        import cv2
        import numpy as np
        
        num_frames = min(duration * fps, 24)
        frames = []
        
        # Create a simple animated background
        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Animated gradient
            phase = i / max(1, num_frames)
            r = int(100 + 155 * (0.5 + 0.5 * np.sin(phase * 2 * np.pi)))
            g = int(100 + 155 * (0.5 + 0.5 * np.sin(phase * 2 * np.pi + 2.094)))
            b = int(100 + 155 * (0.5 + 0.5 * np.sin(phase * 2 * np.pi + 4.188)))
            
            # Gradient background
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
            text = prompt[:30]
            text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y), font, 0.5, (255, 255, 255), 1)
            
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
                'prompt': prompt[:100]
            }
        }
        
    except Exception as e:
        print(f"Fallback video generation failed: {e}")
        return {
            'video_data': '',
            'duration': duration,
            'fps': fps,
            'width': width,
            'height': height,
            'metadata': {'error': str(e)}
        }


def generate_with_loaded_model(
    loaded_model: Dict[str, Any],
    prompt: str,
    context: Dict[str, Any],
    duration_seconds: int,
    fps: int,
    width: int,
    height: int,
    seed: int,
    reference_image: Optional[str]
) -> Dict[str, Any]:
    """
    Generate video using a pre-loaded model.
    This assumes the model is already loaded into VRAM.
    """
    pipeline = loaded_model.get('pipeline')
    device = loaded_model.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    if pipeline is None:
        raise ValueError("Model not properly loaded")
    
    # Check VRAM before generation
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated(0) / (1024**3)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        if vram_used > vram_total * 0.9:
            torch.cuda.empty_cache()
    
    # Generate based on pipeline type
    try:
        if 'stable-video-diffusion' in str(type(pipeline)):
            return _generate_svd(pipeline, prompt, duration_seconds, fps, width, height, seed, reference_image, device)
        else:
            return _generate_diffusion(pipeline, prompt, duration_seconds, fps, width, height, seed, device)
    
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise RuntimeError("Out of VRAM. Try a smaller model or reduce resolution.")


def _generate_svd(pipeline, prompt, duration, fps, width, height, seed, reference_image, device):
    """Generate using Stable Video Diffusion"""
    from diffusers.utils import load_image
    from PIL import Image
    import base64
    from io import BytesIO
    
    # Load reference image
    if reference_image:
        if reference_image.startswith('data:image'):
            img_data = reference_image.split(',')[1]
            image = Image.open(BytesIO(base64.b64decode(img_data)))
        else:
            image = load_image(reference_image)
    else:
        image = _generate_placeholder_image(prompt, width, height)
    
    # Resize to model's expected size
    image = image.resize((width, height))
    
    # Generate
    generator = torch.Generator(device=device).manual_seed(seed)
    
    frames = pipeline(
        image,
        decode_chunk_size=2,
        generator=generator,
        num_frames=min(duration * fps, 24)
    ).frames[0]
    
    video_bytes = _frames_to_video_bytes(frames, fps)
    
    return {
        'video_data': base64.b64encode(video_bytes).decode('utf-8'),
        'duration': duration,
        'fps': fps,
        'width': width,
        'height': height,
        'metadata': {'seed': seed}
    }


def _generate_diffusion(pipeline, prompt, duration, fps, width, height, seed, device):
    """Generate using standard diffusion pipeline"""
    generator = torch.Generator(device=device).manual_seed(seed)
    
    num_frames = min(duration * fps, 24)
    num_inference_steps = 20
    
    frames = pipeline(
        prompt,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
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
        'metadata': {'seed': seed}
    }


def get_available_video_models() -> List[Dict[str, Any]]:
    """Get all video models with their status"""
    return list_video_models()
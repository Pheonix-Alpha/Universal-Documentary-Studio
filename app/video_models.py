"""
Video Models - Enhanced with smart VRAM management
"""

from typing import Dict, Any, Optional, List
import base64
import torch
import numpy as np
from PIL import Image
import tempfile
import os
import cv2

from app import model_manager


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
    device = loaded_model.get('device', 'cuda')
    
    if pipeline is None:
        raise ValueError("Model not properly loaded")
    
    # Check VRAM before generation
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated(0) / (1024**3)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        if vram_used > vram_total * 0.9:
            # VRAM is nearly full - try to clear some
            torch.cuda.empty_cache()
            
            # If still too high, raise warning
            vram_used = torch.cuda.memory_allocated(0) / (1024**3)
            if vram_used > vram_total * 0.85:
                print(f"⚠️ VRAM at {vram_used:.2f}GB / {vram_total:.2f}GB - may cause issues")
    
    # Generate based on pipeline type
    try:
        if 'stable-video-diffusion' in str(type(pipeline)):
            return _generate_svd(pipeline, prompt, duration_seconds, fps, width, height, seed, reference_image, device)
        else:
            return _generate_diffusion(pipeline, prompt, duration_seconds, fps, width, height, seed, device)
    
    except torch.cuda.OutOfMemoryError:
        # Handle OOM gracefully
        torch.cuda.empty_cache()
        raise RuntimeError("Out of VRAM. Try a smaller model or reduce resolution.")


def _generate_svd(pipeline, prompt, duration, fps, width, height, seed, reference_image, device):
    """Generate using Stable Video Diffusion"""
    # Load reference image
    if reference_image:
        import base64
        from io import BytesIO
        if reference_image.startswith('data:image'):
            img_data = reference_image.split(',')[1]
            image = Image.open(BytesIO(base64.b64decode(img_data)))
        else:
            image = Image.open(reference_image)
    else:
        # Generate a placeholder image
        image = Image.new('RGB', (width, height), color='gray')
    
    # Resize to model's expected size
    image = image.resize((width, height))
    
    # Generate
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Use lower decode chunk size for memory efficiency
    frames = pipeline(
        image,
        decode_chunk_size=2,  # Lower = less VRAM usage
        generator=generator,
        num_frames=duration * fps
    ).frames[0]
    
    # Convert to video bytes
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
    
    # Use memory-efficient settings
    num_frames = duration * fps
    num_inference_steps = min(20, 50)  # Reduce steps for memory
    
    # Generate
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


def _frames_to_video_bytes(frames: List[np.ndarray], fps: int) -> bytes:
    """Convert frames to video bytes with memory efficiency"""
    if not frames:
        return b''
    
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


def list_available_video_models() -> List[Dict[str, Any]]:
    """List video models that can be used"""
    models = []
    for model_id, info in model_manager.MODEL_REGISTRY.items():
        if info['type'] == 'video':
            installed = model_manager.is_installed(model_id)
            models.append({
                'id': model_id,
                'name': info['name'],
                'size_gb': info.get('size_gb', 0),
                'installed': installed,
                'description': info['description'],
                'vram_required_gb': info.get('size_gb', 0) * 0.8  # Estimate VRAM needed
            })
    return models
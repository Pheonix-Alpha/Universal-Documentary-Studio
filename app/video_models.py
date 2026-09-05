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

from app import model_manager

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
        return model_manager.is_installed(model_id)
    except Exception:
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

            # Load (or reuse the already-loaded, cached) pipeline through
            # model_manager -- this used to be a completely separate
            # load-from-disk-every-call inside _generate_img2vid/
            # _generate_text2vid below, which re-read the full multi-GB
            # model off disk for EVERY scene in a documentary instead of
            # once per worker session. worker_server.py already calls
            # model_manager.auto_download_and_load_model() right before
            # this, specifically so it's a cheap cache-hit here.
            loaded = model_manager.auto_download_and_load_model(model_id)
            pipe = loaded['pipeline']
            device = loaded['device']

            if model_info['type'] == 'img2vid':
                result = _generate_img2vid(
                    pipe, device, prompt, model_id, context, duration_seconds, fps,
                    width, height, seed, reference_image, callback
                )
            elif model_info['type'] == 'text2vid':
                result = _generate_text2vid(
                    pipe, device, prompt, model_id, context, duration_seconds, fps,
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
    pipe,
    device: str,
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
    """Generate video using Stable Video Diffusion. `pipe`/`device` come
    from model_manager's cache (loaded once per worker session, not
    reloaded from disk here) -- see generate_video() above."""
    try:
        from diffusers.utils import load_image
        
        # Load or create reference image
        if reference_image:
            # NOTE: no `import base64` here -- it's already imported at
            # module level (top of this file). Re-importing it inside this
            # `if` block made Python treat `base64` as a local name for the
            # *whole function* (assignment/import anywhere in a function
            # makes it local throughout), so when reference_image was falsy
            # -- the common case -- `base64` was unbound by the time
            # `base64.b64encode(...)` ran below, raising UnboundLocalError
            # on every real SVD generation without a reference image. That
            # exception was swallowed by generate_video()'s broad
            # try/except, so it silently fell back to the placeholder video
            # instead of ever actually running the real model.
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
    pipe,
    device: str,
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
    """Generate video using text-to-video models. `pipe`/`device` come from
    model_manager's cache -- see generate_video() above. (This also drops
    the old hardcoded `model_path = 'cerspense/zeroscope_v2_576w'` override
    for ZeroScope, which bypassed the locally downloaded copy and would
    have tried to re-download/stream the model straight from the Hub on
    every call.)"""
    try:
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
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
    """Get model path from model_manager. Currently unused internally
    (generate_video()/​_generate_img2vid/_generate_text2vid now get the
    already-loaded pipeline straight from model_manager instead of a path
    to re-load from) but kept as a small public helper in case something
    outside this module wants the on-disk path."""
    try:
        return model_manager.get_model_path(model_id)
    except Exception:
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
    """Convert PIL/NumPy frames to a browser-compatible H.264 MP4."""

    if not frames:
        return b''

    temp_dir = None
    video_path = None

    try:
        import cv2
        import tempfile
        import subprocess
        import shutil

        # --------------------------------------------------
        # Normalize frames to PIL RGB images
        # --------------------------------------------------

        normalized_frames = []

        for frame in frames:

            if isinstance(frame, Image.Image):

                frame = frame.convert("RGB")

            else:

                frame = np.asarray(frame)

                if frame.dtype != np.uint8:
                    frame = np.clip(
                        frame,
                        0,
                        255
                    ).astype(np.uint8)

                if len(frame.shape) == 2:

                    frame = Image.fromarray(
                        frame,
                        mode="L"
                    ).convert("RGB")

                elif frame.shape[2] == 4:

                    frame = Image.fromarray(
                        frame,
                        mode="RGBA"
                    ).convert("RGB")

                else:

                    frame = Image.fromarray(
                        frame
                    ).convert("RGB")

            normalized_frames.append(frame)

        if not normalized_frames:
            return b''

        # --------------------------------------------------
        # Create temporary frame directory
        # --------------------------------------------------

        temp_dir = tempfile.mkdtemp(
            prefix="video_frames_"
        )

        print(
            f"🎞️ Preparing {len(normalized_frames)} frames..."
        )

        # --------------------------------------------------
        # Save frames as PNG
        # --------------------------------------------------

        for i, frame in enumerate(normalized_frames):

            frame_path = os.path.join(
                temp_dir,
                f"frame_{i:04d}.png"
            )

            frame.save(
                frame_path,
                format="PNG"
            )

        # --------------------------------------------------
        # Create temporary MP4
        # --------------------------------------------------

        video_file = tempfile.NamedTemporaryFile(
            suffix=".mp4",
            delete=False
        )

        video_path = video_file.name
        video_file.close()

        # --------------------------------------------------
        # FFmpeg → H.264 + yuv420p
        #
        # This is browser/Colab compatible.
        # --------------------------------------------------

        ffmpeg_command = [
            "ffmpeg",
            "-y",

            "-framerate",
            str(fps),

            "-i",
            os.path.join(
                temp_dir,
                "frame_%04d.png"
            ),

            "-c:v",
            "libx264",

            "-pix_fmt",
            "yuv420p",

            "-movflags",
            "+faststart",

            "-preset",
            "fast",

            "-crf",
            "23",

            video_path
        ]

        print("🎬 Encoding H.264 MP4...")

        process = subprocess.run(
            ffmpeg_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if process.returncode != 0:

            print(
                "❌ FFmpeg encoding failed:"
            )

            print(
                process.stderr[-3000:]
            )

            return b''

        # --------------------------------------------------
        # Read MP4 bytes
        # --------------------------------------------------

        if not os.path.exists(video_path):

            print(
                "❌ FFmpeg did not create video file"
            )

            return b''

        video_size = os.path.getsize(
            video_path
        )

        if video_size == 0:

            print(
                "❌ Generated MP4 is empty"
            )

            return b''

        with open(
            video_path,
            "rb"
        ) as f:

            video_bytes = f.read()

        print(
            f"✅ H.264 MP4 created: "
            f"{len(video_bytes) / 1024:.1f} KB"
        )

        return video_bytes

    except Exception as e:

        print(
            f"❌ Failed to convert frames "
            f"to video: {e}"
        )

        traceback.print_exc()

        return b''

    finally:

        # --------------------------------------------------
        # Cleanup temporary files
        # --------------------------------------------------

        try:

            if temp_dir and os.path.exists(
                temp_dir
            ):

                shutil.rmtree(
                    temp_dir,
                    ignore_errors=True
                )

            if video_path and os.path.exists(
                video_path
            ):

                os.unlink(
                    video_path
                )

        except Exception:
            pass
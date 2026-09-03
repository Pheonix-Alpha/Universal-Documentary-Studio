"""
Model Manager - Complete auto-download, auto-cleanup, auto-swapping
"""

import os
import shutil
import json
import threading
import time
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass, asdict
import hashlib

from huggingface_hub import snapshot_download
import torch

from app import config


@dataclass
class ModelInfo:
    id: str
    name: str
    repo_id: str
    type: str  # 'clip', 'video', 'text'
    size_gb: float
    description: str
    installed: bool = False
    path: str = ""
    last_used: float = 0.0
    in_use: bool = False


# Model Registry with sizes
MODEL_REGISTRY = {
    # Video Models
    'stabilityai/stable-video-diffusion-img2vid': {
        'name': 'Stable Video Diffusion',
        'type': 'video',
        'size_gb': 9.5,
        'description': 'Best quality, image-to-video',
        'vram_gb': 12.0
    },
    'cerspense/zeroscope_v2_576w': {
        'name': 'ZeroScope v2',
        'type': 'video',
        'size_gb': 7.2,
        'description': 'Fast, text-to-video',
        'vram_gb': 8.0
    },
    'SG161222/Realistic_Vision_V5.1_noVAE': {
        'name': 'Realistic Vision',
        'type': 'video',
        'size_gb': 4.8,
        'description': 'Lightweight, good for animation',
        'vram_gb': 6.0
    },
    # CLIP Models (for reference images)
    'openai/clip-vit-base-patch32': {
        'name': 'CLIP ViT-B/32',
        'type': 'clip',
        'size_gb': 0.6,
        'description': 'Base CLIP model',
        'vram_gb': 2.0
    },
    'openai/clip-vit-large-patch14': {
        'name': 'CLIP ViT-L/14',
        'type': 'clip',
        'size_gb': 1.7,
        'description': 'Better CLIP model',
        'vram_gb': 4.0
    }
}

# Maximum storage (Colab: ~70GB, keep 10GB buffer)
MAX_STORAGE_GB = 60.0

# Tracking
_loaded_models = {}
_model_lock = threading.Lock()
_download_in_progress = {}


def get_model_size(model_id: str) -> float:
    """Get model size in GB"""
    info = MODEL_REGISTRY.get(model_id, {})
    return info.get('size_gb', 1.0)


def get_vram_required(model_id: str) -> float:
    """Get VRAM required in GB"""
    info = MODEL_REGISTRY.get(model_id, {})
    return info.get('vram_gb', 4.0)


def is_installed(model_id: str) -> bool:
    """Check if model is installed"""
    model_path = get_model_path(model_id)
    if not os.path.exists(model_path):
        return False
    
    # Check for model files
    required_files = ['model_index.json', 'config.json', 'pytorch_model.bin']
    has_files = False
    for root, dirs, files in os.walk(model_path):
        if any(f in required_files or f.endswith('.bin') or f.endswith('.safetensors') for f in files):
            has_files = True
            break
    
    return has_files


def get_model_path(model_id: str) -> str:
    """Get local model path"""
    safe_id = model_id.replace('/', '__')
    model_dir = os.path.join(config.MODEL_DIR, 'models')
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, safe_id)


def get_installed_models() -> List[str]:
    """Get list of installed model IDs"""
    return [mid for mid in MODEL_REGISTRY if is_installed(mid)]


def list_models() -> List[Dict[str, Any]]:
    """List all models with status"""
    models = []
    for model_id, info in MODEL_REGISTRY.items():
        installed = is_installed(model_id)
        models.append({
            'id': model_id,
            'name': info['name'],
            'type': info['type'],
            'size_gb': info.get('size_gb', 0),
            'vram_gb': info.get('vram_gb', 0),
            'installed': installed,
            'description': info['description'],
            'in_use': _loaded_models.get(model_id) is not None
        })
    return models


def calculate_total_storage() -> float:
    """Calculate total storage used in GB"""
    total = 0.0
    for model_id in MODEL_REGISTRY:
        if is_installed(model_id):
            total += get_actual_model_size_gb(model_id)
    return total


def get_actual_model_size_gb(model_id: str) -> float:
    """Get actual model size in GB"""
    model_path = get_model_path(model_id)
    if not os.path.exists(model_path):
        return 0.0
    
    total_bytes = 0
    for dirpath, dirnames, filenames in os.walk(model_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_bytes += os.path.getsize(fp)
    
    return total_bytes / (1024**3)


def get_available_storage() -> Dict[str, float]:
    """Get storage information"""
    used = calculate_total_storage()
    return {
        'used_gb': used,
        'max_gb': MAX_STORAGE_GB,
        'free_gb': MAX_STORAGE_GB - used,
        'percent_used': (used / MAX_STORAGE_GB) * 100
    }


def get_vram_status() -> Dict[str, Any]:
    """Get VRAM status"""
    if not torch.cuda.is_available():
        return {'device': 'cpu', 'available': False}
    
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    vram_used = torch.cuda.memory_allocated(0) / (1024**3)
    vram_free = vram_total - vram_used
    
    return {
        'device': 'cuda',
        'available': True,
        'total_gb': round(vram_total, 2),
        'used_gb': round(vram_used, 2),
        'free_gb': round(vram_free, 2),
        'percent_used': round((vram_used / vram_total) * 100, 2),
        'loaded_models': [k for k, v in _loaded_models.items() if v is not None]
    }


def auto_download_and_load_model(
    model_id: str,
    device: str = None,
    progress_callback=None
) -> Any:
    """
    AUTO-DOWNLOAD: Downloads model if not installed, then loads into VRAM.
    Automatically cleans up old models to free space.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🔍 Checking model: {model_id}")
    
    # Check if already loaded
    if model_id in _loaded_models and _loaded_models[model_id] is not None:
        print(f"✅ Model already loaded in VRAM")
        return _loaded_models[model_id]
    
    # Check if installed
    if not is_installed(model_id):
        print(f"📥 Model not installed. Downloading...")
        
        # Download with progress
        for pct, msg in smart_download_model(model_id):
            if progress_callback:
                progress_callback(pct, msg)
            print(f"  [{pct:3d}%] {msg}")
        
        print(f"✅ Model downloaded!")
    
    # Load into VRAM (this will unload others if needed)
    with _model_lock:
        # Check VRAM
        if torch.cuda.is_available():
            vram_status = get_vram_status()
            vram_free = vram_status.get('free_gb', 0)
            vram_needed = get_vram_required(model_id)
            
            if vram_free < vram_needed:
                print(f"⚠️ Not enough VRAM. Unloading old models...")
                _unload_unused_models(keep=[])
                torch.cuda.empty_cache()
        
        # Load the model
        loaded = _load_model_into_vram(model_id, device)
        _loaded_models[model_id] = loaded
        _update_last_used(model_id)
        
        print(f"✅ Model loaded into VRAM")
        return loaded


def smart_download_model(
    model_id: str,
    force: bool = False,
    progress_callback=None
) -> Generator[tuple, None, None]:
    """
    Smart download with automatic cleanup of old models.
    Yields (progress_percent, message) tuples.
    """
    if model_id not in MODEL_REGISTRY:
        yield (0, f"❌ Unknown model: {model_id}")
        return
    
    # Check if already downloading
    if model_id in _download_in_progress and _download_in_progress[model_id]:
        yield (0, f"⏳ Download already in progress")
        return
    
    model_info = MODEL_REGISTRY[model_id]
    model_size = model_info.get('size_gb', 1.0)
    model_path = get_model_path(model_id)
    
    # Check if already installed
    if is_installed(model_id) and not force:
        yield (100, f"✅ {model_info['name']} already installed")
        return
    
    # Mark as downloading
    _download_in_progress[model_id] = True
    
    try:
        # Calculate available space
        current_usage = calculate_total_storage()
        available = MAX_STORAGE_GB - current_usage
        
        yield (5, f"📊 Storage: {current_usage:.2f}GB / {MAX_STORAGE_GB}GB")
        
        # Check if we need to free space
        if model_size > available:
            needed = model_size - available + 1.0
            yield (10, f"⚠️ Need {needed:.2f}GB more space. Cleaning old models...")
            
            # Free space by deleting least recently used models
            freed = _smart_cleanup(needed)
            
            if freed < needed:
                yield (50, f"❌ Cannot free enough space. Freed {freed:.2f}GB, needed {needed:.2f}GB")
                _download_in_progress[model_id] = False
                return
            
            current_usage = calculate_total_storage()
            available = MAX_STORAGE_GB - current_usage
            yield (20, f"✅ Freed {freed:.2f}GB. Available: {available:.2f}GB")
        
        # Download the model
        yield (30, f"⬇️ Downloading {model_info['name']} ({model_size:.1f}GB)...")
        
        # Download
        # NOTE: this used to pass ignore_patterns=["*.safetensors", "*.bin"]
        # for CLIP models, which excluded BOTH possible weight formats and
        # left nothing but config files on disk -- is_installed() would then
        # incorrectly report the model as installed (config.json is enough
        # to pass that check) and loading would fail with a missing-weights
        # error the first time the model was actually used. Download
        # everything; these CLIP checkpoints are small (<2GB).
        snapshot_download(
            repo_id=model_id,
            local_dir=model_path,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4,
        )
        
        yield (90, f"✅ Downloaded {model_info['name']}")
        
        # Update last used
        _update_last_used(model_id)
        
        yield (100, f"✅ {model_info['name']} ready!")
        
    except Exception as e:
        yield (0, f"❌ Download failed: {e}")
        # Clean up partial download
        if os.path.exists(model_path):
            shutil.rmtree(model_path, ignore_errors=True)
    
    finally:
        _download_in_progress[model_id] = False


def _smart_cleanup(needed_gb: float) -> float:
    """
    Intelligently delete models to free up space.
    Deletes: not in use > oldest > largest
    """
    freed = 0.0
    
    # Get all installed models
    installed_models = []
    for model_id in MODEL_REGISTRY:
        if is_installed(model_id):
            size = get_actual_model_size_gb(model_id)
            last_used = _get_last_used(model_id)
            in_use = _loaded_models.get(model_id) is not None
            
            installed_models.append({
                'id': model_id,
                'size': size,
                'last_used': last_used,
                'in_use': in_use,
                'type': MODEL_REGISTRY[model_id]['type']
            })
    
    # Sort: not in use first, then oldest, then largest
    installed_models.sort(key=lambda x: (
        0 if not x['in_use'] else 1,  # Not in use first
        -x['last_used'],  # Older first
        -x['size']  # Larger first
    ))
    
    # Delete until we have enough space
    deleted = []
    for model in installed_models:
        if freed >= needed_gb:
            break
        
        # Don't delete models in use
        if model['in_use']:
            continue
        
        # Delete the model
        try:
            model_path = get_model_path(model['id'])
            if os.path.exists(model_path):
                shutil.rmtree(model_path, ignore_errors=True)
                freed += model['size']
                deleted.append(model['id'])
                
                # Remove from loaded models if present
                if model['id'] in _loaded_models:
                    del _loaded_models[model['id']]
        except Exception as e:
            print(f"Failed to delete {model['id']}: {e}")
    
    if deleted:
        print(f"🧹 Deleted models to free space: {', '.join(deleted)}")
    
    return freed


def _load_model_into_vram(model_id: str, device: str) -> Any:
    """Load model into VRAM"""
    model_info = MODEL_REGISTRY.get(model_id)
    if not model_info:
        raise ValueError(f"Unknown model: {model_id}")
    
    if model_info['type'] == 'video':
        return _load_video_model(model_id, device)
    elif model_info['type'] == 'clip':
        return _load_clip_model(model_id, device)
    else:
        raise ValueError(f"Unknown model type: {model_info['type']}")


def _load_video_model(model_id: str, device: str) -> Dict[str, Any]:
    """Load video model"""
    from diffusers import DiffusionPipeline
    
    model_path = get_model_path(model_id)
    if not os.path.exists(model_path):
        raise ValueError(f"Model not downloaded: {model_id}")
    
    dtype = torch.float16 if device == 'cuda' else torch.float32
    
    # Special handling for different models
    if 'stable-video-diffusion' in model_id:
        from diffusers import StableVideoDiffusionPipeline
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            variant="fp16" if device == 'cuda' else None,
            low_cpu_mem_usage=True
        )
    else:
        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
    
    # Enable optimizations
    if device == 'cuda':
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()
    
    pipe.to(device)
    pipe.eval()
    
    return {'pipeline': pipe, 'device': device}


def _load_clip_model(model_id: str, device: str) -> Dict[str, Any]:
    """Load CLIP model"""
    from transformers import CLIPModel, CLIPProcessor
    
    model_path = get_model_path(model_id)
    dtype = torch.float16 if device == 'cuda' else torch.float32
    
    model = CLIPModel.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()
    
    processor = CLIPProcessor.from_pretrained(model_path)
    
    return {'model': model, 'processor': processor, 'device': device}


def _unload_unused_models(keep: List[str] = None):
    """Unload models from VRAM to free memory"""
    global _loaded_models
    keep = keep or []

    for model_id, model_obj in list(_loaded_models.items()):
        if model_id not in keep and model_obj is not None:
            # Move to CPU. Loaded models are dicts like {'pipeline': ..., 'device': ...}
            # or {'model': ..., 'processor': ..., 'device': ...} -- not the raw
            # model object itself -- so reach into them rather than calling
            # .to() on the wrapper dict (which has no such method).
            for key in ('pipeline', 'model'):
                inner = model_obj.get(key) if isinstance(model_obj, dict) else None
                if inner is not None and hasattr(inner, 'to'):
                    try:
                        inner.to('cpu')
                    except Exception:
                        pass

            _loaded_models[model_id] = None
            print(f"🧹 Unloaded {model_id} from VRAM")

    # Clear from GPU once after moving everything off it, not per-model.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Clean up dict (was previously reassigning the module-level name without
    # `global`, which raised UnboundLocalError the moment this ran).
    _loaded_models = {k: v for k, v in _loaded_models.items() if v is not None}


def _update_last_used(model_id: str):
    """Update last used timestamp"""
    timestamp_file = os.path.join(get_model_path(model_id), '.last_used')
    os.makedirs(os.path.dirname(timestamp_file), exist_ok=True)
    with open(timestamp_file, 'w') as f:
        json.dump({'timestamp': time.time()}, f)


def _get_last_used(model_id: str) -> float:
    """Get last used timestamp"""
    timestamp_file = os.path.join(get_model_path(model_id), '.last_used')
    if os.path.exists(timestamp_file):
        try:
            with open(timestamp_file, 'r') as f:
                data = json.load(f)
                return data.get('timestamp', 0)
        except:
            pass
    return 0


def delete_model(model_id: str) -> bool:
    """Delete a model. Unloads it from VRAM first (if loaded) so we don't
    leave a dangling pipeline object pointing at files that no longer
    exist -- this is also what lets the "switch models" flow (delete the
    old one, then install what's actually needed) work cleanly."""
    if model_id in _loaded_models:
        _unload_unused_models(keep=[m for m in _loaded_models if m != model_id])
        _loaded_models.pop(model_id, None)

    model_path = get_model_path(model_id)
    if os.path.exists(model_path):
        shutil.rmtree(model_path, ignore_errors=True)
        if model_id in _download_in_progress:
            del _download_in_progress[model_id]
        return True
    return False
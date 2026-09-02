"""
Model Manager - Enhanced with VRAM and storage optimization
Handles automatic model swapping to maximize resources
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

# Import config
from app import config


@dataclass
class ModelInfo:
    """Information about a model"""
    id: str
    name: str
    repo_id: str
    type: str  # 'clip', 'video', 'text'
    size_gb: float
    description: str
    installed: bool = False
    path: str = ""
    last_used: float = 0.0
    priority: int = 0  # Higher = more likely to keep


# Enhanced Model Registry with size information
MODEL_REGISTRY = {
    # CLIP Models (for image ranking)
    'openai/clip-vit-base-patch32': {
        'name': 'CLIP ViT-B/32',
        'type': 'clip',
        'size_gb': 0.6,
        'description': 'Base CLIP model for image-text matching'
    },
    'openai/clip-vit-large-patch14': {
        'name': 'CLIP ViT-L/14',
        'type': 'clip',
        'size_gb': 1.7,
        'description': 'Larger CLIP model for better matching'
    },
    'laion/CLIP-ViT-H-14-laion2B-s32B-b79K': {
        'name': 'CLIP ViT-H/14 (LAION)',
        'type': 'clip',
        'size_gb': 3.5,
        'description': 'High-quality CLIP model'
    },
    
    # Video Generation Models
    'stabilityai/stable-video-diffusion-img2vid': {
        'name': 'Stable Video Diffusion',
        'type': 'video',
        'size_gb': 9.5,
        'description': 'Image-to-video generation'
    },
    'cerspense/zeroscope_v2_576w': {
        'name': 'ZeroScope v2',
        'type': 'video',
        'size_gb': 7.2,
        'description': 'Text-to-video generation'
    },
    'damo-vilab/modelscope-damo-text-to-video-synthesis': {
        'name': 'ModelScope Text2Video',
        'type': 'video',
        'size_gb': 11.3,
        'description': 'Text-to-video synthesis'
    },
    'SG161222/Realistic_Vision_V5.1_noVAE': {
        'name': 'Realistic Vision (AnimateDiff)',
        'type': 'video',
        'size_gb': 4.8,
        'description': 'AnimateDiff base model'
    },
    
    # Text Models
    'sentence-transformers/all-MiniLM-L6-v2': {
        'name': 'Sentence Transformer (MiniLM)',
        'type': 'text',
        'size_gb': 0.1,
        'description': 'Lightweight text embeddings'
    }
}

# Maximum storage we can use (Colab free ~70GB, keep 10GB buffer)
MAX_STORAGE_GB = 60.0

# Memory tracking
_loaded_models = {}  # model_id -> loaded state
_model_lock = threading.Lock()
_storage_usage = {}  # model_id -> size in bytes


def get_model_size_gb(model_id: str) -> float:
    """Get model size in GB"""
    info = MODEL_REGISTRY.get(model_id)
    if info:
        return info.get('size_gb', 1.0)
    return 1.0


def get_installed_models() -> List[str]:
    """Get list of installed model IDs"""
    installed = []
    for model_id in MODEL_REGISTRY:
        if is_installed(model_id):
            installed.append(model_id)
    return installed


def get_installed_models_with_size() -> List[Dict[str, Any]]:
    """Get installed models with their sizes"""
    models = []
    total_size = 0
    
    for model_id, info in MODEL_REGISTRY.items():
        if is_installed(model_id):
            size = get_actual_model_size(model_id)
            models.append({
                'id': model_id,
                'name': info['name'],
                'type': info['type'],
                'size_gb': size / (1024**3),
                'path': get_model_path(model_id),
                'last_used': _get_model_last_used(model_id)
            })
            total_size += size
    
    return models, total_size


def get_actual_model_size(model_id: str) -> int:
    """Get actual model size in bytes"""
    model_path = get_model_path(model_id)
    if not os.path.exists(model_path):
        return 0
    
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(model_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    
    # Cache the size
    _storage_usage[model_id] = total_size
    return total_size


def calculate_total_storage_used() -> float:
    """Calculate total storage used by all models in GB"""
    total = 0.0
    for model_id in MODEL_REGISTRY:
        if is_installed(model_id):
            total += get_actual_model_size(model_id) / (1024**3)
    return total


def smart_download_model(
    model_id: str,
    force: bool = False,
    progress_callback=None
) -> Generator[tuple, None, None]:
    """
    Smart download with automatic cleanup of old models.
    Yields (progress_percent, message) tuples.
    """
    if not model_id or model_id not in MODEL_REGISTRY:
        yield (0, f"❌ Unknown model: {model_id}")
        return
    
    model_info = MODEL_REGISTRY[model_id]
    model_size_gb = model_info.get('size_gb', 1.0)
    model_path = get_model_path(model_id)
    
    # Check if already installed
    if is_installed(model_id) and not force:
        yield (100, f"✅ {model_info['name']} already installed")
        return
    
    # Calculate available space
    current_usage = calculate_total_storage_used()
    available = MAX_STORAGE_GB - current_usage
    
    yield (5, f"📊 Current storage: {current_usage:.2f}GB / {MAX_STORAGE_GB}GB")
    
    # Check if we need to free space
    if model_size_gb > available:
        yield (10, f"⚠️ Need {model_size_gb - available:.2f}GB more space. Cleaning old models...")
        
        # Free space by deleting least recently used models
        freed_space = _smart_cleanup(model_size_gb - available + 1.0)  # +1GB buffer
        
        if freed_space < (model_size_gb - available):
            yield (50, f"❌ Cannot free enough space. Needed {model_size_gb - available:.2f}GB, freed {freed_space:.2f}GB")
            return
        
        current_usage = calculate_total_storage_used()
        available = MAX_STORAGE_GB - current_usage
        yield (20, f"✅ Freed {freed_space:.2f}GB. Available: {available:.2f}GB")
    
    # Download the model
    try:
        yield (30, f"⬇️ Downloading {model_info['name']} ({model_size_gb:.1f}GB)...")
        
        # Download
        snapshot_download(
            repo_id=model_id,
            local_dir=model_path,
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.safetensors", "*.bin"] if model_info['type'] == 'clip' else [],
            max_workers=4
        )
        
        yield (90, f"✅ Downloaded {model_info['name']}")
        
        # Update last used
        _update_model_last_used(model_id)
        
        yield (100, f"✅ {model_info['name']} ready!")
        
    except Exception as e:
        yield (0, f"❌ Download failed: {e}")
        # Clean up partial download
        if os.path.exists(model_path):
            shutil.rmtree(model_path, ignore_errors=True)


def _smart_cleanup(needed_gb: float) -> float:
    """
    Intelligently delete models to free up space.
    Prioritizes: unused > older > smaller models
    Returns: amount of space freed in GB
    """
    freed = 0.0
    
    # Get all installed models with their last used times
    installed_models = []
    for model_id in MODEL_REGISTRY:
        if is_installed(model_id):
            size = get_actual_model_size(model_id) / (1024**3)
            last_used = _get_model_last_used(model_id)
            installed_models.append({
                'id': model_id,
                'size': size,
                'last_used': last_used,
                'type': MODEL_REGISTRY[model_id]['type']
            })
    
    # Sort by: never used first, then oldest, then largest
    installed_models.sort(key=lambda x: (
        0 if x['last_used'] == 0 else 1,  # Never used first
        -x['last_used'],  # Older first
        -x['size']  # Larger first
    ))
    
    # Delete models until we have enough space
    deleted = []
    for model in installed_models:
        if freed >= needed_gb:
            break
        
        # Don't delete currently loaded models
        if model['id'] in _loaded_models and _loaded_models[model['id']]:
            continue
        
        # Delete the model
        try:
            model_path = get_model_path(model['id'])
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
                freed += model['size']
                deleted.append(model['id'])
        except Exception as e:
            print(f"Failed to delete {model['id']}: {e}")
    
    if deleted:
        print(f"🧹 Deleted models to free space: {', '.join(deleted)}")
    
    return freed


def _update_model_last_used(model_id: str):
    """Update the last used timestamp for a model"""
    timestamp_file = os.path.join(get_model_path(model_id), '.last_used')
    with open(timestamp_file, 'w') as f:
        json.dump({'timestamp': time.time()}, f)


def _get_model_last_used(model_id: str) -> float:
    """Get the last used timestamp for a model"""
    timestamp_file = os.path.join(get_model_path(model_id), '.last_used')
    if os.path.exists(timestamp_file):
        try:
            with open(timestamp_file, 'r') as f:
                data = json.load(f)
                return data.get('timestamp', 0)
        except:
            pass
    return 0


def load_model_into_vram(model_id: str, device: str = None) -> Any:
    """
    Load a model into VRAM with automatic unloading of other models.
    This is the key function for VRAM management.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Check if already loaded
    if model_id in _loaded_models and _loaded_models[model_id]:
        # Update last used
        _update_model_last_used(model_id)
        return _loaded_models[model_id]
    
    with _model_lock:
        # Unload other models to free VRAM
        _unload_unused_models(keep=[model_id])
        
        # Load the model
        model_info = MODEL_REGISTRY.get(model_id)
        if not model_info:
            raise ValueError(f"Unknown model: {model_id}")
        
        # Load based on type
        loaded_model = None
        if model_info['type'] == 'clip':
            loaded_model = _load_clip_model(model_id, device)
        elif model_info['type'] == 'video':
            loaded_model = _load_video_model(model_id, device)
        elif model_info['type'] == 'text':
            loaded_model = _load_text_model(model_id, device)
        
        # Store loaded model
        _loaded_models[model_id] = loaded_model
        _update_model_last_used(model_id)
        
        return loaded_model


def _unload_unused_models(keep: List[str] = None):
    """
    Unload models from VRAM to free memory.
    Keeps specified models loaded.
    """
    keep = keep or []
    
    for model_id, model_obj in list(_loaded_models.items()):
        if model_id not in keep and model_obj is not None:
            # Different ways to unload based on model type
            if hasattr(model_obj, 'to'):
                try:
                    model_obj.to('cpu')
                except:
                    pass
            
            # Delete from GPU
            if hasattr(model_obj, 'cuda'):
                try:
                    model_obj.cpu()
                except:
                    pass
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Remove reference
            _loaded_models[model_id] = None
            print(f"🧹 Unloaded {model_id} from VRAM")
    
    # Clean up the dict
    _loaded_models = {k: v for k, v in _loaded_models.items() if v is not None}


def _load_clip_model(model_id: str, device: str):
    """Load a CLIP model into VRAM"""
    from transformers import CLIPModel, CLIPProcessor
    
    model_path = get_model_path(model_id)
    if not os.path.exists(model_path):
        raise ValueError(f"Model not downloaded: {model_id}")
    
    # Load with memory optimization
    model = CLIPModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()
    
    processor = CLIPProcessor.from_pretrained(model_path)
    
    return {'model': model, 'processor': processor, 'device': device}


def _load_video_model(model_id: str, device: str):
    """Load a video generation model into VRAM"""
    from diffusers import DiffusionPipeline
    
    model_path = get_model_path(model_id)
    if not os.path.exists(model_path):
        raise ValueError(f"Model not downloaded: {model_id}")
    
    # Different loading strategies for different video models
    if 'stable-video-diffusion' in model_id:
        from diffusers import StableVideoDiffusionPipeline
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            variant="fp16" if device == 'cuda' else None,
            low_cpu_mem_usage=True
        )
    else:
        # Generic diffusion pipeline
        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            low_cpu_mem_usage=True
        )
    
    # Enable optimizations
    if device == 'cuda':
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()
    
    pipe.to(device)
    
    return {'pipeline': pipe, 'device': device}


def _load_text_model(model_id: str, device: str):
    """Load a text model into VRAM"""
    from sentence_transformers import SentenceTransformer
    
    model_path = get_model_path(model_id)
    model = SentenceTransformer(model_path, device=device)
    return {'model': model, 'device': device}


# ---- Basic functions kept from original ----

def is_installed(model_id: str) -> bool:
    """Check if a model is installed"""
    model_path = get_model_path(model_id)
    if not os.path.exists(model_path):
        return False
    
    # Check for required files
    required_files = ['model_index.json', 'config.json', 'pytorch_model.bin']
    # Some models use different file structures
    has_files = False
    for root, dirs, files in os.walk(model_path):
        if any(f in required_files or f.endswith('.bin') or f.endswith('.safetensors') for f in files):
            has_files = True
            break
    
    return has_files


def get_model_path(model_id: str) -> str:
    """Get the local path for a model"""
    safe_id = model_id.replace('/', '__')
    model_dir = os.path.join(config.MODEL_DIR, 'models')
    return os.path.join(model_dir, safe_id)


def list_models() -> List[Dict[str, Any]]:
    """List all models with their status"""
    models = []
    for model_id, info in MODEL_REGISTRY.items():
        installed = is_installed(model_id)
        size = get_actual_model_size(model_id) / (1024**3) if installed else info.get('size_gb', 0)
        models.append({
            'id': model_id,
            'name': info['name'],
            'type': info['type'],
            'size_gb': round(size, 2),
            'installed': installed,
            'description': info['description']
        })
    return models


def delete_model(model_id: str) -> bool:
    """Delete a model"""
    model_path = get_model_path(model_id)
    if os.path.exists(model_path):
        shutil.rmtree(model_path, ignore_errors=True)
        # Clear from loaded models
        if model_id in _loaded_models:
            del _loaded_models[model_id]
        return True
    return False


def get_available_storage() -> Dict[str, float]:
    """Get storage information"""
    current_usage = calculate_total_storage_used()
    return {
        'used_gb': current_usage,
        'max_gb': MAX_STORAGE_GB,
        'free_gb': MAX_STORAGE_GB - current_usage,
        'percent_used': (current_usage / MAX_STORAGE_GB) * 100
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
        'loaded_models': list(_loaded_models.keys())
    }
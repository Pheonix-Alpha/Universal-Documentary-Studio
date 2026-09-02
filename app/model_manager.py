"""
Model Manager - Fixed video model loading for all pipeline types
"""

import os
import shutil
import json
import threading
import time
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass, asdict

from huggingface_hub import snapshot_download
import torch

from app import config


@dataclass
class ModelInfo:
    id: str
    name: str
    repo_id: str
    type: str
    size_gb: float
    description: str
    installed: bool = False
    path: str = ""
    last_used: float = 0.0
    in_use: bool = False


# Model Registry
MODEL_REGISTRY = {
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

MAX_STORAGE_GB = 60.0

_loaded_models = {}
_model_lock = threading.Lock()
_download_in_progress = {}


def get_model_size(model_id: str) -> float:
    info = MODEL_REGISTRY.get(model_id, {})
    return info.get('size_gb', 1.0)


def get_vram_required(model_id: str) -> float:
    info = MODEL_REGISTRY.get(model_id, {})
    return info.get('vram_gb', 4.0)


def is_installed(model_id: str) -> bool:
    """Check if model is installed"""
    model_path = get_model_path(model_id)
    if not os.path.exists(model_path):
        return False
    
    for root, dirs, files in os.walk(model_path):
        for f in files:
            if f.endswith('.bin') or f.endswith('.safetensors') or f == 'model_index.json':
                return True
    return False


def get_model_path(model_id: str) -> str:
    safe_id = model_id.replace('/', '__')
    model_dir = os.path.join(config.MODEL_DIR, 'models')
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, safe_id)


def get_installed_models() -> List[str]:
    return [mid for mid in MODEL_REGISTRY if is_installed(mid)]


def list_models() -> List[Dict[str, Any]]:
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
    total = 0.0
    for model_id in MODEL_REGISTRY:
        if is_installed(model_id):
            total += get_actual_model_size_gb(model_id)
    return total


def get_actual_model_size_gb(model_id: str) -> float:
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
    used = calculate_total_storage()
    return {
        'used_gb': used,
        'max_gb': MAX_STORAGE_GB,
        'free_gb': MAX_STORAGE_GB - used,
        'percent_used': (used / MAX_STORAGE_GB) * 100
    }


def get_vram_status() -> Dict[str, Any]:
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
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🔍 Checking model: {model_id}")
    
    if model_id in _loaded_models and _loaded_models[model_id] is not None:
        print(f"✅ Model already loaded in VRAM")
        return _loaded_models[model_id]
    
    if not is_installed(model_id):
        print(f"📥 Model not installed. Downloading...")
        
        for pct, msg in smart_download_model(model_id):
            if progress_callback:
                progress_callback(pct, msg)
            print(f"  [{pct:3d}%] {msg}")
        
        print(f"✅ Model downloaded!")
    
    with _model_lock:
        if torch.cuda.is_available():
            vram_status = get_vram_status()
            vram_free = vram_status.get('free_gb', 0)
            vram_needed = get_vram_required(model_id)
            
            if vram_free < vram_needed:
                print(f"⚠️ Not enough VRAM. Unloading old models...")
                _unload_unused_models(keep=[])
                torch.cuda.empty_cache()
        
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
    if model_id not in MODEL_REGISTRY:
        yield (0, f"❌ Unknown model: {model_id}")
        return
    
    if model_id in _download_in_progress and _download_in_progress[model_id]:
        yield (0, f"⏳ Download already in progress")
        return
    
    model_info = MODEL_REGISTRY[model_id]
    model_size = model_info.get('size_gb', 1.0)
    model_path = get_model_path(model_id)
    
    if is_installed(model_id) and not force:
        yield (100, f"✅ {model_info['name']} already installed")
        return
    
    _download_in_progress[model_id] = True
    
    try:
        current_usage = calculate_total_storage()
        available = MAX_STORAGE_GB - current_usage
        
        yield (5, f"📊 Storage: {current_usage:.2f}GB / {MAX_STORAGE_GB}GB")
        
        if model_size > available:
            needed = model_size - available + 1.0
            yield (10, f"⚠️ Need {needed:.2f}GB more space. Cleaning old models...")
            
            freed = _smart_cleanup(needed)
            
            if freed < needed:
                yield (50, f"❌ Cannot free enough space. Freed {freed:.2f}GB, needed {needed:.2f}GB")
                _download_in_progress[model_id] = False
                return
            
            current_usage = calculate_total_storage()
            available = MAX_STORAGE_GB - current_usage
            yield (20, f"✅ Freed {freed:.2f}GB. Available: {available:.2f}GB")
        
        yield (30, f"⬇️ Downloading {model_info['name']} ({model_size:.1f}GB)...")
        
        snapshot_download(
            repo_id=model_id,
            local_dir=model_path,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4
        )
        
        yield (90, f"✅ Downloaded {model_info['name']}")
        _update_last_used(model_id)
        yield (100, f"✅ {model_info['name']} ready!")
        
    except Exception as e:
        yield (0, f"❌ Download failed: {e}")
        if os.path.exists(model_path):
            shutil.rmtree(model_path, ignore_errors=True)
    
    finally:
        _download_in_progress[model_id] = False


def _smart_cleanup(needed_gb: float) -> float:
    freed = 0.0
    
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
    
    installed_models.sort(key=lambda x: (
        0 if not x['in_use'] else 1,
        -x['last_used'],
        -x['size']
    ))
    
    deleted = []
    for model in installed_models:
        if freed >= needed_gb:
            break
        
        if model['in_use']:
            continue
        
        try:
            model_path = get_model_path(model['id'])
            if os.path.exists(model_path):
                shutil.rmtree(model_path, ignore_errors=True)
                freed += model['size']
                deleted.append(model['id'])
                
                if model['id'] in _loaded_models:
                    del _loaded_models[model['id']]
        except Exception as e:
            print(f"Failed to delete {model['id']}: {e}")
    
    if deleted:
        print(f"🧹 Deleted models: {', '.join(deleted)}")
    
    return freed


def _load_model_into_vram(model_id: str, device: str) -> Any:
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
    """
    Load video model - FIXED: Handles all pipeline types properly
    """
    from diffusers import DiffusionPipeline
    
    model_path = get_model_path(model_id)
    if not os.path.exists(model_path):
        raise ValueError(f"Model not downloaded: {model_id}")
    
    print(f"📦 Loading video model: {model_id}")
    print(f"   Device: {device}")
    
    dtype = torch.float16 if device == 'cuda' else torch.float32
    
    # Special handling for different models
    try:
        if 'stable-video-diffusion' in model_id:
            from diffusers import StableVideoDiffusionPipeline
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                model_path,
                dtype=dtype,
                low_cpu_mem_usage=True,
                variant="fp16" if device == 'cuda' else None
            )
        else:
            # For ZeroScope and other models
            pipe = DiffusionPipeline.from_pretrained(
                model_path,
                dtype=dtype,
                low_cpu_mem_usage=True
            )
        
        # FIXED: Check if pipeline has eval() before calling it
        if hasattr(pipe, 'eval'):
            pipe.eval()
        
        # Handle device placement
        if device == 'cuda':
            try:
                # Try CPU offloading first (memory efficient)
                if hasattr(pipe, 'enable_model_cpu_offload'):
                    pipe.enable_model_cpu_offload()
                    print("   Using CPU offloading (memory efficient)")
                else:
                    # Fallback: move to device directly
                    pipe.to(device)
                    print(f"   Moved to {device}")
            except Exception as e:
                print(f"   Offloading failed, trying direct device placement...")
                try:
                    pipe.to(device)
                    print(f"   Moved to {device}")
                except Exception as e2:
                    print(f"   Device placement failed: {e2}")
                    # Keep on CPU
                    print("   Keeping on CPU")
        else:
            # CPU mode
            try:
                pipe.to('cpu')
            except:
                pass
            print("   Using CPU mode")
        
        return {'pipeline': pipe, 'device': device}
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        # Fallback: try loading without optimization
        try:
            print("   Attempting fallback load without optimizations...")
            pipe = DiffusionPipeline.from_pretrained(
                model_path,
                low_cpu_mem_usage=True
            )
            
            if hasattr(pipe, 'eval'):
                pipe.eval()
            
            # Try to move to device
            try:
                pipe.to(device)
            except:
                pass
            
            return {'pipeline': pipe, 'device': device}
        except Exception as e2:
            raise RuntimeError(f"Both loading attempts failed: {e2}")


def _load_clip_model(model_id: str, device: str) -> Dict[str, Any]:
    from transformers import CLIPModel, CLIPProcessor
    
    model_path = get_model_path(model_id)
    dtype = torch.float16 if device == 'cuda' else torch.float32
    
    model = CLIPModel.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    model.to(device)
    
    if hasattr(model, 'eval'):
        model.eval()
    
    processor = CLIPProcessor.from_pretrained(model_path)
    
    return {'model': model, 'processor': processor, 'device': device}


def _unload_unused_models(keep: List[str] = None):
    keep = keep or []
    
    for model_id, model_obj in list(_loaded_models.items()):
        if model_id not in keep and model_obj is not None:
            try:
                # Try to move to CPU
                if hasattr(model_obj, 'to'):
                    try:
                        model_obj.to('cpu')
                    except:
                        pass
            except:
                pass
            
            _loaded_models[model_id] = None
            print(f"🧹 Unloaded {model_id}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    _loaded_models = {k: v for k, v in _loaded_models.items() if v is not None}


def _update_last_used(model_id: str):
    model_path = get_model_path(model_id)
    timestamp_file = os.path.join(model_path, '.last_used')
    os.makedirs(os.path.dirname(timestamp_file), exist_ok=True)
    with open(timestamp_file, 'w') as f:
        json.dump({'timestamp': time.time()}, f)


def _get_last_used(model_id: str) -> float:
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
    model_path = get_model_path(model_id)
    if os.path.exists(model_path):
        shutil.rmtree(model_path, ignore_errors=True)
        if model_id in _loaded_models:
            del _loaded_models[model_id]
        if model_id in _download_in_progress:
            del _download_in_progress[model_id]
        return True
    return False
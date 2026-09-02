"""
Worker Client - Handles communication with GPU workers
"""

from typing import List, Dict, Any, Optional
import requests
import time
import json
import base64
from io import BytesIO
from PIL import Image
import itertools

# Worker registry
_workers = {}
_rr_counter = itertools.count()


def add_worker(url: str, label: str = None) -> str:
    """Add a worker to the registry"""
    # Normalize URL
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = f'https://{url}'
    url = url.rstrip('/')
    
    # Check if already exists
    for worker_id, w in _workers.items():
        if w['url'] == url:
            return worker_id
    
    # Create new worker entry
    worker_id = f"worker_{int(time.time())}"
    _workers[worker_id] = {
        'id': worker_id,
        'url': url,
        'label': label or url,
        'added_at': time.time(),
        'last_health_check': 0,
        'connected': False,
        'status': 'unknown',
        'device': 'unknown',
        'capabilities': {},
        'load': 0
    }
    
    # Initial health check
    check_worker(worker_id)
    
    return worker_id


def remove_worker(worker_id: str) -> bool:
    """Remove a worker from the registry"""
    if worker_id in _workers:
        del _workers[worker_id]
        return True
    return False


def check_worker(worker_id: str) -> tuple:
    """Check a worker's health and update its status"""
    worker = _workers.get(worker_id)
    if not worker:
        return False, "Worker not found"
    
    try:
        response = requests.get(f"{worker['url']}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            worker['connected'] = True
            worker['status'] = data.get('status', 'ok')
            worker['device'] = data.get('device', 'unknown')
            worker['capabilities'] = data.get('capabilities', {})
            worker['last_health_check'] = time.time()
            return True, "Connected"
    except Exception as e:
        print(f"Worker health check failed: {e}")
    
    worker['connected'] = False
    worker['status'] = 'disconnected'
    return False, "Disconnected"


def list_workers() -> List[Dict[str, Any]]:
    """List all workers with current status"""
    workers = []
    for worker_id, worker in _workers.items():
        connected, status = check_worker(worker_id)
        workers.append({
            'id': worker_id,
            'url': worker['url'],
            'label': worker['label'],
            'connected': connected,
            'status': status,
            'device': worker.get('device', 'unknown'),
            'capabilities': worker.get('capabilities', {}),
            'load': worker.get('load', 0)
        })
    return workers


def is_any_connected() -> bool:
    """Check if any workers are connected"""
    for worker_id in list(_workers.keys()):
        connected, _ = check_worker(worker_id)
        if connected:
            return True
    return False


def connected_worker_ids() -> List[str]:
    """Get all connected worker IDs"""
    connected = []
    for worker_id in list(_workers.keys()):
        connected_worker, _ = check_worker(worker_id)
        if connected_worker:
            connected.append(worker_id)
    return connected


def get_worker(worker_id: str) -> Optional[Dict[str, Any]]:
    """Get worker by ID"""
    return _workers.get(worker_id)


# ---- Video Generation Functions ----

def generate_video_on_worker(
    worker_id: str,
    prompt: str,
    model_id: str,
    context: Dict[str, Any] = None,
    duration_seconds: int = 4,
    fps: int = 24,
    width: int = 576,
    height: int = 320,
    seed: int = 42,
    reference_image: Optional[str] = None,
    timeout: int = 300
) -> Dict[str, Any]:
    """Call a worker to generate video"""
    worker = _workers.get(worker_id)
    if not worker:
        raise ValueError(f"Worker {worker_id} not found")
    
    if not worker.get('connected', False):
        raise RuntimeError(f"Worker {worker_id} is not connected")
    
    # Increment load
    worker['load'] = worker.get('load', 0) + 1
    
    try:
        payload = {
            'prompt': prompt,
            'model_id': model_id,
            'context': context or {},
            'duration_seconds': duration_seconds,
            'fps': fps,
            'width': width,
            'height': height,
            'seed': seed,
            'reference_image': reference_image
        }
        
        response = requests.post(
            f"{worker['url']}/video/generate",
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        
        result = response.json()
        
        if 'video_data' not in result:
            raise RuntimeError("Worker returned invalid response")
        
        return result
        
    finally:
        worker['load'] = max(0, worker.get('load', 0) - 1)


def generate_video_round_robin(
    prompt: str,
    model_id: str,
    context: Dict[str, Any] = None,
    duration_seconds: int = 4,
    fps: int = 24,
    width: int = 576,
    height: int = 320,
    seed: int = 42,
    reference_image: Optional[str] = None,
    max_retries: int = 3
) -> Dict[str, Any]:
    """Generate video using workers in round-robin fashion"""
    connected = connected_worker_ids()
    if not connected:
        raise RuntimeError("No connected workers available")
    
    # Try each worker
    for attempt in range(max_retries):
        # Pick next worker
        worker_id = connected[next(_rr_counter) % len(connected)]
        
        try:
            return generate_video_on_worker(
                worker_id, prompt, model_id, context,
                duration_seconds, fps, width, height,
                seed, reference_image
            )
        except Exception as e:
            print(f"Worker {worker_id} failed: {e}")
            # Remove from connected and try next
            if worker_id in connected:
                connected.remove(worker_id)
            
            if not connected:
                break
            
            continue
    
    raise RuntimeError("All workers failed to generate video")


def list_video_models_on_worker(worker_id: str) -> List[Dict[str, Any]]:
    """List video models available on a specific worker"""
    worker = _workers.get(worker_id)
    if not worker:
        return []
    
    try:
        response = requests.get(f"{worker['url']}/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('models', [])
    except Exception:
        pass
    return []
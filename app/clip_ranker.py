"""Loads a locally-downloaded CLIP model and ranks candidate images against scene text."""
import io

import requests
import torch
from PIL import Image

from app import model_manager as mm

_loaded = {}


def _load_clip(model_id: str = "clip-vit-b-32"):
    if model_id in _loaded:
        return _loaded[model_id]
    if not mm.is_installed(model_id):
        raise RuntimeError(f"Model '{model_id}' is not downloaded yet. Use the Models tab to download it.")

    from transformers import CLIPModel, CLIPProcessor

    path = mm.get_model_path(model_id)
    model = CLIPModel.from_pretrained(path)
    processor = CLIPProcessor.from_pretrained(path)
    model.eval()
    _loaded[model_id] = (model, processor)
    return model, processor


def _fetch_image(url: str, timeout: int = 8):
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "research-ai-storyboard/1.0"})
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


def rank_candidates(text: str, candidates: list, model_id: str = "clip-vit-b-32", top_k: int = 5):
    """Returns the top_k candidates, each with an added 'score' (cosine similarity, -1..1)
    and 'image' (PIL.Image) field. Candidates whose image can't be fetched are dropped."""
    model, processor = _load_clip(model_id)

    text_inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_emb = model.get_text_features(**text_inputs)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    scored = []
    for c in candidates:
        img = _fetch_image(c["url"])
        if img is None:
            continue
        img_inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            img_emb = model.get_image_features(**img_inputs)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            sim = (text_emb @ img_emb.T).item()
        c2 = dict(c)
        c2["score"] = round(sim, 4)
        c2["image"] = img
        scored.append(c2)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

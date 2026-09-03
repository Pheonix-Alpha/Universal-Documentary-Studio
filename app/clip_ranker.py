"""Loads a locally-downloaded CLIP model and ranks candidate media against scene text."""
import io

import requests
import torch
from PIL import Image

from app import model_manager as mm

_loaded = {}


DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"  # must match a key in model_manager.MODEL_REGISTRY


def _load_clip(model_id: str = DEFAULT_CLIP_MODEL_ID):
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


def _extract_embedding(output, *primary_attrs):
    """model.get_text_features()/get_image_features() normally return a plain
    tensor, but some transformers versions/model variants wrap the result in
    an output object instead. Unwrap defensively rather than assuming either
    shape -- this is what was causing AttributeError on `.norm()`."""
    if isinstance(output, torch.Tensor):
        return output
    for attr in primary_attrs:
        if hasattr(output, attr):
            return getattr(output, attr)
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0, :]  # CLS token fallback
    raise TypeError(f"Could not extract an embedding tensor from {type(output)}")


def rank_candidates(text: str, candidates: list, model_id: str = DEFAULT_CLIP_MODEL_ID, top_k: int = 5):
    """Returns the top_k candidates, each with an added 'score' (cosine
    similarity, -1..1) and 'image' (PIL.Image, used for both CLIP scoring
    and gallery/thumbnail display). Video candidates are scored via their
    'thumbnail_url' rather than the (unplayable-as-an-image) 'url'.
    Candidates whose thumbnail can't be fetched are dropped."""
    model, processor = _load_clip(model_id)

    text_inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_emb = _extract_embedding(model.get_text_features(**text_inputs), "text_embeds")
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    scored = []
    for c in candidates:
        thumb_url = c.get("thumbnail_url") or c.get("url")
        img = _fetch_image(thumb_url)
        if img is None:
            continue
        img_inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            img_emb = _extract_embedding(model.get_image_features(**img_inputs), "image_embeds")
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            sim = (text_emb @ img_emb.T).item()
        c2 = dict(c)
        c2["score"] = round(sim, 4)
        c2["image"] = img
        scored.append(c2)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
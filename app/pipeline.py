"""
Orchestrates: story -> scenes -> queries -> multi-source retrieval -> CLIP ranking.

run_pipeline() is a generator so the Gradio UI can stream live status +
percentage + partial gallery results as each stage completes.
"""
from app import clip_ranker, image_sources, query_generator, scene_analyzer
from app import model_manager as mm


def run_pipeline(story: str, clip_model_id: str = "clip-vit-b-32", top_k: int = 3):
    if not story or not story.strip():
        yield {"stage": "Idle", "pct": 0, "log": "Enter a story and click Start.", "gallery": []}
        return

    if not mm.is_installed(clip_model_id):
        yield {
            "stage": "Error",
            "pct": 0,
            "log": f"Model '{clip_model_id}' isn't downloaded yet. Go to the Models tab and download it first.",
            "gallery": [],
        }
        return

    yield {"stage": "Scene extraction", "pct": 3, "log": "Analyzing story into scenes...", "gallery": []}
    scenes = scene_analyzer.analyze_story(story)
    yield {"stage": "Scene extraction", "pct": 12, "log": f"Found {len(scenes)} scene(s).", "gallery": []}

    all_results = []
    n = max(len(scenes), 1)

    for i, scene in enumerate(scenes):
        base_pct = 12 + int(80 * i / n)
        desc_preview = scene.get("description", "")[:70]

        yield {
            "stage": "Query generation",
            "pct": base_pct + 2,
            "log": f"Scene {i + 1}/{n}: '{desc_preview}...' -> generating search queries",
            "gallery": all_results,
        }
        queries = query_generator.generate_queries(scene)

        yield {
            "stage": "Image retrieval",
            "pct": base_pct + 6,
            "log": f"Scene {i + 1}/{n}: searching Wikimedia / Unsplash / web for {len(queries)} quer{'y' if len(queries)==1 else 'ies'}...",
            "gallery": all_results,
        }
        candidates = []
        for q in queries:
            candidates += image_sources.gather_candidates(q)
        seen, unique = set(), []
        for c in candidates:
            if c["url"] not in seen:
                seen.add(c["url"])
                unique.append(c)

        yield {
            "stage": "CLIP ranking",
            "pct": base_pct + 12,
            "log": f"Scene {i + 1}/{n}: ranking {len(unique)} candidate image(s) with {clip_model_id}...",
            "gallery": all_results,
        }
        ranked = clip_ranker.rank_candidates(scene.get("description", ""), unique, model_id=clip_model_id, top_k=top_k)

        for r in ranked:
            caption = f"Scene {scene.get('scene_id', i + 1)} | score={r['score']} | {r['source']}"
            all_results.append((r["image"], caption))

        best = ranked[0]["score"] if ranked else "N/A"
        yield {
            "stage": "CLIP ranking",
            "pct": base_pct + int(80 / n),
            "log": f"Scene {i + 1}/{n}: done -- best score {best}, kept {len(ranked)} image(s).",
            "gallery": all_results,
        }

    yield {"stage": "Finished", "pct": 100, "log": "Storyboard complete.", "gallery": all_results}

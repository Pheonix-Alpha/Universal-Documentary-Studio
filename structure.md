# Project Structure & Execution Flow (Updated)

> This replaces the previous `structure.md`, which described a planned/older
> version of the system (Claude-API-powered bible generation, a separate
> "outputs/" tree, a 5-tab Gradio UI, etc.). The sections below were
> regenerated directly from the current `main` branch source (commit
> `b88b307`), function-by-function, so every file/class/function name here
> actually exists in the repo today.

## 🎯 What this repo actually is right now

This is **not yet** the multi-agent "Universal Documentary Studio" described
in the README's aspirational sections. The current codebase is a
**story → scene breakdown → production bible → distributed video-clip
generation** pipeline, wrapped in a single Gradio app, with an optional
FastAPI "worker" process that does the actual GPU-heavy generation. There is
no `adapters/`, `agents/`, `core/`, `engines/`, `qa/`, or `tests/` directory
in this branch — the entire implementation lives under `app/`, driven by two
entry points (`launcher.py`, `worker.py`).

Key architectural fact: **there is no Anthropic/Claude API dependency
anymore.** Story analysis, production-bible generation, and script
enhancement all run on a local "brain" LLM (`Qwen/Qwen2.5-7B-Instruct`,
4-bit quantized) loaded on the main Colab's own GPU via
`model_manager.generate_text()`, with rule-based fallbacks if that model
isn't available. `config.py` no longer has an Anthropic key at all — only
image-source API keys (Unsplash, Flickr, Pexels, Pixabay).

```
┌─────────────────────────────────────────────────────────────────────┐
│                     MAIN COLAB — "the director"                     │
│  (Gradio app: launcher.py → app/gradio_app.py)                      │
├─────────────────────────────────────────────────────────────────────┤
│  📖 Local "brain" LLM (Qwen2.5-7B, 4-bit)                            │
│     scene_analyzer → production_bible → script_enhancer             │
│                                                                       │
│  🎬 scene_orchestrator → ProductionUnit list                        │
│  🧩 consistency_manager → per-unit worker context (seeds/style)     │
│  🧠 model_selector → auto-picks a video model for the job/worker    │
│  🌐 worker_client → sends generation jobs to connected workers      │
│  🖼️ compute.py / clip_ranker.py → local CLIP fallback ranking       │
└─────────────────────────────────────────────────────────────────────┘
                                   │  HTTP (requests)
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  WORKER COLAB(S) — "render nodes"                   │
│  (worker.py → app/worker_server.py, FastAPI + cloudflared tunnel)   │
├─────────────────────────────────────────────────────────────────────┤
│  /health /models /models/{id}/download /models/{id}/progress        │
│  /models/storage /models/cleanup /rank /video/generate              │
│     → model_manager.py (download/VRAM mgmt) + video_models.py       │
└─────────────────────────────────────────────────────────────────────┘
```

## 📁 Complete file structure (actual)

```
Universal-Documentary-Studio/
├── launcher.py              # MAIN entry point (installs deps, launches Gradio)
├── worker.py                # WORKER entry point (FastAPI server + cloudflared tunnel)
├── requirements.txt
├── README.md                # Describes the earlier "storyboard" version of the project
├── BUGFIXES.md              # Log of crash fixes vs. the pre-fix main branch
├── structure.md             # (this file — regenerated)
│
└── app/
    ├── __init__.py                  # empty
    ├── config.py                    # Paths (MODEL_DIR/DATA_DIR) + image-source API keys
    │   ├── KEY_SPECS                # Unsplash / Flickr / Pexels / Pixabay specs
    │   ├── get_key() / set_key() / is_key_set()
    │
    ├── model_manager.py             # Unified model registry + smart download/VRAM mgmt (653 lines)
    │   ├── ModelInfo                 (dataclass)
    │   ├── MODEL_REGISTRY            # video (SVD, ZeroScope, Realistic Vision),
    │   │                             #   CLIP (ViT-B/32, ViT-L/14), and the local
    │   │                             #   LLM (Qwen2.5-7B-Instruct) in ONE registry
    │   ├── LOCAL_BRAIN_MODEL_ID      # = 'Qwen/Qwen2.5-7B-Instruct'
    │   ├── get_model_size() / get_vram_required() / is_installed() / get_model_path()
    │   ├── get_installed_models() / list_models()
    │   ├── calculate_total_storage() / get_actual_model_size_gb() / get_available_storage()
    │   ├── get_vram_status()
    │   ├── auto_download_and_load_model()
    │   ├── smart_download_model()    # yields (pct, msg) progress
    │   ├── _smart_cleanup()          # frees space by evicting least-used models
    │   ├── _load_model_into_vram() → _load_video_model() / _load_clip_model() / _load_llm_model()
    │   ├── generate_text()           # runs the local brain LLM for JSON-structured tasks
    │   ├── _unload_unused_models() / _update_last_used() / _get_last_used()
    │   └── delete_model()
    │
    ├── model_selector.py            # "The brain decides which video model to use" (NEW)
    │   ├── VIDEO_MODEL_PRIORITY     # SVD > ZeroScope > Realistic Vision
    │   ├── _worker_health()         # reads worker's /health for VRAM/storage
    │   ├── select_video_model()     # picks best model for job + worker resources
    │   └── describe_selection()
    │
    ├── clip_ranker.py               # Local CLIP text↔image similarity ranking
    │   ├── _load_clip() / _fetch_image() / _extract_embedding()
    │   └── rank_candidates()
    │
    ├── compute.py                   # Dispatch: local CLIP vs. worker CLIP
    │   ├── backend_name() / list_models() / is_installed()
    │   └── rank_candidates()        # routes to clip_ranker or worker_client
    │
    ├── video_models.py              # Video generation interface (396 lines)
    │   ├── list_video_models() / is_video_model_installed()
    │   ├── generate_video()         # main entry (img2vid or text2vid based on model)
    │   ├── _generate_img2vid() / _generate_text2vid()
    │   ├── generate_fallback()      # public wrapper (added per BUGFIXES.md #4)
    │   ├── _generate_fallback_video() # placeholder-image based fallback, no GPU needed
    │   ├── _get_model_path() / _create_placeholder_image()
    │   └── _frames_to_video_bytes()
    │
    ├── scene_analyzer.py            # Story → scenes
    │   ├── _fallback_scene_split()  # rule-based sentence splitter
    │   ├── _strip_code_fence()
    │   ├── _analyze_with_local_brain()   # uses model_manager.generate_text()
    │   └── analyze_story()          # entry point, tries local brain then falls back
    │
    ├── query_generator.py           # Scene → search queries (for reference images)
    │   ├── _fallback_queries() / _generate_with_local_brain()
    │   └── generate_queries()
    │
    ├── image_sources.py             # Multi-source reference-image retrieval (445 lines)
    │   ├── _candidate()
    │   ├── search_wikimedia() / search_nasa() / search_internet_archive()
    │   ├── search_met_museum() / search_openverse() / search_loc()
    │   ├── search_flickr() / search_pexels() / search_pixabay()
    │   ├── _get_ddgs_class() / search_duckduckgo()
    │   └── gather_candidates()      # fans out across all sources
    │
    ├── production_bible.py          # Character/location/style bible generation (321 lines)
    │   ├── _scene_people()          # accepts scene['people'] as list OR comma-string
    │   ├── Character / Location / VisualStyle / ProductionBible  (dataclasses)
    │   ├── ProductionBible.to_dict()
    │   ├── generate_production_bible()      # entry point
    │   ├── _generate_bible_with_local_brain()  # uses model_manager.generate_text()
    │   ├── _generate_bible_fallback()       # rule-based
    │   └── _enrich_scenes_with_bible()
    │
    ├── script_enhancer.py           # Dialogue/narrative improvement
    │   ├── enhance_script()         # entry point
    │   ├── _enhance_with_local_brain()
    │   └── _enhance_fallback()
    │
    ├── scene_orchestrator.py        # Bible → per-scene production units
    │   ├── ProductionUnit           (dataclass) + to_dict()
    │   └── SceneOrchestrator
    │       ├── breakdown() / _create_unit() / _generate_visual_prompt()
    │       ├── get_pending_units() / get_assigned_units()
    │       ├── update_unit_status() / get_unit_by_id()
    │
    ├── consistency_manager.py       # Cross-worker visual/character consistency
    │   └── ConsistencyManager
    │       ├── get_worker_context()          # seeds, style, character/location data
    │       ├── get_consistency_hash()
    │       ├── share_character_reference()
    │       └── get_shared_character_reference()
    │
    ├── worker_client.py             # Main Colab's HTTP client to worker(s) (337 lines)
    │   ├── add_worker() / remove_worker() / check_worker() / list_workers()
    │   ├── is_any_connected() / connected_worker_ids() / get_worker()
    │   ├── generate_video_on_worker()
    │   ├── generate_video_round_robin()
    │   ├── list_video_models_on_worker() / list_models()
    │   ├── rank_candidates_on_worker() / rank_candidates_round_robin()   # added, see BUGFIXES.md #3
    │   └── switch_video_model()     # deletes other resident video model, installs the chosen one
    │
    ├── worker_server.py             # FastAPI app run on the worker (392 lines)
    │   ├── RankRequest / VideoGenRequest / VideoGenResponse   (Pydantic models)
    │   ├── build_fastapi_app()
    │   │   ├── GET  /health                 # VRAM/storage status
    │   │   ├── GET  /models                 # list with install status
    │   │   ├── POST /models/{id}/download   # smart download w/ progress
    │   │   ├── GET  /models/{id}/progress
    │   │   ├── GET  /models/storage
    │   │   ├── POST /models/cleanup
    │   │   ├── POST /rank                   # CLIP ranking
    │   │   └── POST /video/generate
    │   └── _generate_with_model()
    │
    ├── pipeline.py                  # Orchestrates the whole flow, yields UI progress (263 lines)
    │   └── run_video_pipeline(story, model_id="auto", duration_per_scene, auto_download,
    │                           fps, width, height, quality_preference)
    │       ├── Stage 1: Validate story (≥ 20 chars)
    │       ├── Stage 2: Check for connected workers
    │       ├── Stage 2b: Ensure local brain LLM is downloaded/loaded
    │       ├── Stage 3: production_bible.generate_production_bible() + script_enhancer.enhance_script()
    │       │            (re-runs scene analysis if the enhanced script actually changed)
    │       ├── Stage 4: scene_orchestrator.breakdown() → ProductionUnits
    │       ├── Stage 4b: model_selector.select_video_model() ("auto" by default)
    │       │            → worker_client.switch_video_model() if a worker is connected
    │       ├── Stage 5: per-scene generation loop
    │       │            → worker_client.generate_video_on_worker() (with consistency context)
    │       │            → or video_models.generate_fallback() if no worker is connected
    │       └── Stage 6: yield finished clips + first clip as "final_video"
    │
    └── gradio_app.py                # UI (420 lines)
        ├── _b64_video_to_tempfile()
        └── build_app()
            ├── Tab 1: "🎬 Generate"   — story box, video-model dropdown (auto/manual),
            │                            duration slider, progress/log, gallery + final video,
            │                            collapsible "Model Status (Auto)" JSON panel
            ├── Tab 2: "⚙️ Workers"    — add/remove/list connected GPU workers
            └── Tab 3: "🔑 API Keys"   — set the image-source keys from config.KEY_SPECS
        └── launch_app()
```

> Note: the older `structure.md` listed an `outputs/{bibles,clips,final}`
> directory and a 5-tab UI including a dedicated "Models & Resources"
> dashboard tab and an "Image Sources" tab. Neither exists in the current
> `gradio_app.py` (3 tabs only) or on disk (no `outputs/` directory is
> created by any current code path — generated clip bytes are kept
> in-memory/base64 and streamed straight to the Gradio gallery/video
> components).

## 🔄 Execution flow

### 1. Main app — `python launcher.py`

```
launcher.py
    ├── install_requirements()     # pip install -r requirements.txt
    ├── launch_app()
    │   └── gradio_app.build_app() → demo.launch(share=True)
    └── main()                     # CLI entry point
```

### 2. Worker — `python worker.py`

```
worker.py
    ├── install_requirements()
    ├── download_cloudflared()     # fetches the cloudflared binary
    ├── start_server()             # runs worker_server's FastAPI app (in a thread)
    ├── start_tunnel(port=8000)    # opens a cloudflared quick tunnel, prints the public URL
    └── main()
```

The printed `https://xxxx.trycloudflare.com` URL is pasted into the main
app's "⚙️ Workers" tab via `worker_client.add_worker()`.

### 3. Generate-video flow — clicking "🎬 Generate Video"

```
gradio_app.build_app()  (Tab 1 callback)
    │
    ▼
pipeline.run_video_pipeline(story, model_id, duration_per_scene, ...)
    │
    ├── validate story
    ├── worker_client.is_any_connected()
    ├── model_manager.smart_download_model(LOCAL_BRAIN_MODEL_ID)   # first run only
    │
    ├── production_bible.generate_production_bible(story)
    │   ├── scene_analyzer.analyze_story(story)
    │   │   └── _analyze_with_local_brain()  → model_manager.generate_text()
    │   ├── _generate_bible_with_local_brain() / _generate_bible_fallback()
    │   └── _enrich_scenes_with_bible()
    │
    ├── script_enhancer.enhance_script(story, bible)
    │   └── (re-analyzes scenes if the script actually changed)
    │
    ├── scene_orchestrator.SceneOrchestrator(bible).breakdown()
    │   └── _generate_visual_prompt() per scene → ProductionUnit list
    │
    ├── consistency_manager.ConsistencyManager(bible)
    │
    ├── model_selector.select_video_model(units, worker_id, quality_preference)
    │   └── reads worker /health, picks from VIDEO_MODEL_PRIORITY
    ├── worker_client.switch_video_model(worker_id, chosen_model_id)
    │
    └── for each ProductionUnit:
            worker_client.generate_video_on_worker(
                worker_id, prompt=unit.visual_prompt, model_id=chosen_model_id,
                context=consistency.get_worker_context(unit),
                duration_seconds, fps, width, height, seed=unit.seed
            )
            # OR, if no worker connected:
            video_models.generate_fallback(prompt, duration, fps, width, height, seed)
```

### 4. Worker-side video generation — `POST /video/generate`

```
worker_server.build_fastapi_app()  → /video/generate
    └── _generate_with_model(model_id, prompt, duration, fps, width, height, seed)
        ├── model_manager.auto_download_and_load_model(model_id)  # download if needed
        │   └── smart_download_model() → _smart_cleanup() if storage is tight
        ├── model_manager._load_model_into_vram() → _load_video_model()
        │   └── _unload_unused_models() first if VRAM is tight
        └── video_models.generate_video(loaded_model, prompt, ...)
            ├── _generate_img2vid()  or  _generate_text2vid()   (by model type)
            └── _frames_to_video_bytes(frames, fps)             # → base64 MP4 bytes
```

## 🗺️ File dependency map

```
launcher.py
    └── app/gradio_app.py
        ├── app/pipeline.py
        │   ├── app/production_bible.py
        │   │   ├── app/scene_analyzer.py
        │   │   ├── app/script_enhancer.py
        │   │   └── app/model_manager.py   (generate_text — local brain)
        │   ├── app/scene_orchestrator.py  → app/production_bible.py
        │   ├── app/consistency_manager.py → app/production_bible.py
        │   ├── app/model_selector.py      → app/model_manager.py, app/worker_client.py
        │   ├── app/worker_client.py       → app/compute.py
        │   ├── app/video_models.py        → app/model_manager.py
        │   └── app/model_manager.py
        └── (Workers tab) → app/worker_client.py
            (API Keys tab) → app/config.py

worker.py
    └── app/worker_server.py
        ├── app/model_manager.py
        ├── app/video_models.py
        ├── app/clip_ranker.py
        └── app/config.py

app/compute.py
    ├── app/clip_ranker.py     (local CLIP path)
    └── app/worker_client.py   (remote CLIP path)

app/image_sources.py           # used by an earlier storyboard flow (query_generator +
app/query_generator.py         # image_sources); not currently called from pipeline.py,
                                # kept for the reference-image retrieval feature described
                                # in README.md
```

## 🔧 Key data structures

**`ProductionUnit`** (`scene_orchestrator.py`)
```python
{
    'scene_id': int,
    'description': str,
    'characters': List[str],
    'location': str,
    'visual_prompt': str,
    'duration_seconds': int,
    'seed': int,
    'style_context': Dict,
    'assigned_worker': Optional[str],
    'status': str,               # pending | generating | completed | failed
}
```

**`MODEL_REGISTRY` entry** (`model_manager.py`)
```python
{
    'name': str,
    'type': str,          # 'video' | 'clip' | 'llm'
    'size_gb': float,
    'description': str,
    'vram_gb': float,
}
```

**Worker context** (`consistency_manager.get_worker_context()`)
```python
{
    'global_seed': int,
    'scene_seed': int,
    'style': {...},        # from ProductionBible.VisualStyle
    'characters': {name: {...}},
    'locations': {name: {...}},
}
```

## 🎯 Where to make common changes

| I want to...                                  | Edit this |
|------------------------------------------------|-----------|
| Add/remove a video, CLIP, or LLM model          | `model_manager.py → MODEL_REGISTRY` |
| Change the local-brain model                    | `model_manager.py → LOCAL_BRAIN_MODEL_ID` |
| Change storage/VRAM limits or cleanup strategy  | `model_manager.py → MAX_STORAGE_GB`, `_smart_cleanup()` |
| Change auto model-selection logic               | `model_selector.py → VIDEO_MODEL_PRIORITY`, `select_video_model()` |
| Change how scenes are split                     | `scene_analyzer.py → analyze_story()` |
| Change bible generation                         | `production_bible.py → generate_production_bible()` |
| Change script enhancement                       | `script_enhancer.py → enhance_script()` |
| Change per-scene visual prompts                 | `scene_orchestrator.py → _generate_visual_prompt()` |
| Change cross-scene consistency rules            | `consistency_manager.py → get_worker_context()` |
| Change worker load balancing                    | `worker_client.py → generate_video_round_robin()` |
| Add a worker API endpoint                       | `worker_server.py → build_fastapi_app()` |
| Change the pipeline flow/stages                 | `pipeline.py → run_video_pipeline()` |
| Add a new image source                          | `image_sources.py` (add a `search_*()` + wire into `gather_candidates()`) |
| Add a new image-source API key                  | `config.py → KEY_SPECS` |
| Change the UI                                   | `gradio_app.py → build_app()` |

## 🚀 Deployment commands

**Main Colab:**
```bash
git clone <YOUR_REPO_URL> Universal-Documentary-Studio
cd Universal-Documentary-Studio
python launcher.py
# wait for the printed Gradio URL (https://xxxx.gradio.live)
```

**Worker Colab (one or more):**
```bash
git clone <YOUR_REPO_URL> Universal-Documentary-Studio
cd Universal-Documentary-Studio
python worker.py
# wait for the printed cloudflared URL (https://xxxx.trycloudflare.com)
# paste it into the main app's "⚙️ Workers" tab
```

## 📊 Monitoring resources

Via the worker's own API:
```bash
curl http://<worker-url>/health           # VRAM + storage status
curl http://<worker-url>/models           # installed models + status
curl http://<worker-url>/models/storage   # storage breakdown
```

## ⚠️ Known gaps vs. the README's stated goal

- `README.md` still describes this as a "story → verified image storyboard"
  prototype, and the top-level narration in some prior planning docs
  describes a much larger "Universal Documentary Studio" (per-day
  documentaries, Shorts, thumbnails, human-review dashboard, `ARCHITECTURE.md`,
  `MODELS.md`, `LICENSES.md`, etc.). None of that scaffolding
  (`adapters/`, `agents/`, `core/`, `engines/`, `qa/`, `tests/`,
  `ARCHITECTURE.md`, `MODELS.md`, `LICENSES.md`, `SETUP_COLAB.md`,
  `TROUBLESHOOTING.md`) exists in the current `main` branch — this file
  documents what is actually implemented today, not that longer-term plan.
- `image_sources.py` / `query_generator.py` (multi-source reference-image
  retrieval + CLIP ranking) are fully implemented but not currently wired
  into `pipeline.run_video_pipeline()` — they're reachable via
  `compute.py`/`clip_ranker.py` but the video pipeline doesn't call them.
- There is no automated test suite (`pytest`, `tests/`) in this branch.

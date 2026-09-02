# Project Structure & Execution Flow

This document maps every file and function in the codebase, and traces the
order things actually run in, so future changes are easy to place. There
are two independent entry points:

- **`launcher.py`** — the MAIN app (story → storyboard UI). Run this in
  your primary Colab notebook (CPU or GPU, either works).
- **`worker.py`** — the OPTIONAL GPU worker. Run this in one or more
  *separate* Colab notebooks (each with a GPU runtime) when you want CLIP
  ranking offloaded instead of run locally.

They share the `app/` package but never import each other directly — they
only talk over HTTP once a worker is connected from the main app's UI.

```
Universal-Documentary-Studio/
├── launcher.py          # MAIN entry point
├── worker.py             # WORKER entry point (separate Colab)
├── requirements.txt
└── app/
    ├── config.py          # settings + API keys (read live, not at import)
    ├── model_manager.py   # local model download/list/delete (disk cache)
    ├── clip_ranker.py     # loads a local CLIP model, scores candidates
    ├── compute.py         # picks local vs worker(s) for every run
    ├── worker_client.py   # HTTP client the MAIN app uses to call workers
    ├── worker_server.py   # FastAPI app that the WORKER exposes
    ├── scene_analyzer.py  # story -> scenes
    ├── query_generator.py # scene -> search queries
    ├── image_sources.py   # queries -> candidate images/videos (many APIs)
    ├── pipeline.py         # orchestrates the whole story -> storyboard run
    └── gradio_app.py       # the UI (tabs, buttons, wiring)
```

---

## 1. MAIN app flow — `python launcher.py`

### `launcher.py`
The only thing run directly by the user for the main app.

- **`install_requirements()`** — runs `pip install -r requirements.txt` as
  a subprocess so a fresh Colab needs exactly one command. Exits the
  process if pip fails.
- **`install_requirements()` is called first, then `launch_app()`:**
- **`launch_app()`** — adds the repo root to `sys.path`, imports
  `app.gradio_app.build_app()`, builds the Gradio `Blocks` UI, and calls
  `demo.queue().launch(share=True)`. `share=True` is required because
  Colab doesn't expose `localhost` to your browser directly — it gives you
  a public `*.gradio.live` URL instead.
- `if __name__ == "__main__":` → `install_requirements()` → `launch_app()`

From here on, everything is event-driven inside the Gradio app — nothing
else runs top-to-bottom until a button is clicked.

### `app/gradio_app.py` — the UI shell
**`build_app()`** constructs one `gr.Blocks` with four tabs. It is the hub
that wires every other module to a button/click:

- **Tab "Generate Storyboard"**
  - Inputs: story text box, CLIP model dropdown (`_clip_model_choices()`),
    top-k slider, Start button.
  - Outputs: stage label, progress bar, log box, image gallery.
  - `_clip_model_choices()` — calls `compute.list_models()` and returns
    just the CLIP model ids, so the dropdown always reflects whichever
    backend (local or worker) is currently active.
  - `_run(story, model_id, top_k)` — the Start button's click handler.
    Iterates `pipeline.run_pipeline(...)` (a generator) and yields each
    `(stage, pct, log, gallery)` update straight into the UI, so progress
    streams live instead of blocking until the whole story is done.

- **Tab "Models"** (local disk cache, always available)
  - `render_models(_tick)` — re-runs every time `refresh_state` changes.
    Lists every model from `model_manager.list_models()`; for each one
    shows a Download button (streams `model_manager.download_model_stream()`
    into a progress slider) or a Delete button
    (`model_manager.delete_model()`), then bumps `refresh_state` to force
    a re-render.

- **Tab "⚙️ Workers (GPU offload)"** (zero or more remote GPUs)
  - `_add_worker(url, label)` — calls `worker_client.add_worker()`,
    registering a new worker connection.
  - `render_workers(_tick)` — calls `worker_client.list_workers()` (which
    pings every registered worker's `/health`), and for each one renders
    an Accordion with a Remove button and, if connected, that worker's own
    model list with Download/Delete buttons hitting
    `worker_client.download_model_stream()` / `worker_client.delete_model()`
    for that specific worker id.

- **Tab "🔑 API Keys"**
  - `render_keys(_tick)` — iterates `config.KEY_SPECS`; for each key shows
    connected/not-set status, a password-masked input, and Save/Clear
    buttons that call `config.set_key()` and immediately re-render.

### The pipeline — `app/pipeline.py`
**`run_pipeline(story, clip_model_id, top_k)`** is a generator; every
`yield` is one UI update. This is the real "flow" of a single run:

1. Guard clause: empty story → yield an "Idle" message and stop.
2. `compute.is_installed(clip_model_id)` — if the model isn't ready on the
   active backend, yield an "Error" stage explaining where to download it,
   and stop.
3. `scene_analyzer.analyze_story(story)` → list of scene dicts.
4. For each scene, in order:
   a. `query_generator.generate_queries(scene)` → 3-4 search strings.
   b. `image_sources.gather_candidates(query, use_duckduckgo=...)` for
      each query (DuckDuckGo is capped to the first
      `MAX_DDG_QUERIES_PER_SCENE` queries per scene, with a
      `DELAY_BETWEEN_WEB_QUERIES_SEC` pause between web calls, because its
      unofficial endpoint rate-limits hard). Results are de-duplicated by
      URL.
   c. `compute.rank_candidates(scene_description, candidates, model_id, top_k)`
      → the `top_k` best-matching images/videos for that scene.
   d. Each ranked result is appended to the running gallery and yielded.
5. Final yield: stage "Finished", pct 100, full gallery.

### The dispatcher — `app/compute.py`
This is the "check in the background whether a worker is present" layer
that `pipeline.py` and `gradio_app.py` call instead of talking to
`model_manager`/`clip_ranker`/`worker_client` directly:

- **`backend_name()`** — `"worker"` if `worker_client.is_any_connected()`
  is true (i.e. at least one registered worker answers `/health` right
  now), else `"local"`. Called fresh every time — there's no caching, so a
  worker dropping mid-session is noticed on the very next call.
- **`list_models()`** — if a worker backend is active, asks the first
  connected worker for its registry (all workers run the same code, so any
  one of them is representative); otherwise asks `model_manager`.
- **`is_installed(model_id)`** — on the worker backend, requires the model
  to be installed on **every** connected worker (since ranking will
  round-robin across all of them); on local, checks `model_manager`.
- **`rank_candidates(...)`** — worker backend → 
  `worker_client.rank_candidates_round_robin(...)`; local backend →
  `clip_ranker.rank_candidates(...)`.

### Local model management — `app/model_manager.py`
Tracks what's downloaded under `./models/<model_id>/`.

- **`MODEL_REGISTRY`** — dict of the 3 known CLIP variants (id → name,
  HF repo id, size, description). Add a new model here to make it
  available everywhere (local Models tab, worker registry, dropdown).
- **`is_installed(model_id)`** — folder exists and isn't empty.
- **`get_model_path(model_id)`** — local folder path, passed to
  `transformers` to load the model.
- **`list_models()`** — every registry entry + its `installed` flag; this
  is the shape both the local Models tab and the worker's `/models`
  endpoint return.
- **`delete_model(model_id)`** — `shutil.rmtree` the model folder.
- **`_download_blocking(model_id, progress_cb)`** — calls
  `huggingface_hub.snapshot_download`, with a `tqdm` subclass
  (`ProgressTqdm`) that calls `progress_cb(pct, msg)` on every chunk so
  progress can be streamed.
- **`download_model_stream(model_id)`** — generator wrapper: runs
  `_download_blocking` on a background thread, relays `(pct, msg)` tuples
  through a `queue.Queue` so the caller can `for pct, msg in ...:` from
  the main thread (this is what both the Gradio click handler and the
  worker's background job iterate).

### Local ranking — `app/clip_ranker.py`
Only used when `compute.backend_name() == "local"` (no worker connected).

- **`_load_clip(model_id)`** — loads and caches (`_loaded` dict) a
  `CLIPModel` + `CLIPProcessor` from the local model folder
  (`model_manager.get_model_path`). Raises if not downloaded yet.
- **`_fetch_image(url)`** — downloads a candidate's thumbnail as a PIL
  image; returns `None` on any failure (dead link, timeout, etc).
- **`_extract_embedding(output, *attrs)`** — defensive unwrapping, since
  different `transformers` versions/model variants return either a plain
  tensor or a wrapped output object from `get_text_features()` /
  `get_image_features()`.
- **`rank_candidates(text, candidates, model_id, top_k)`** — embeds the
  scene text once, then for each candidate: fetches its thumbnail, embeds
  it, cosine-similarity scores it against the text, and keeps the
  `top_k` highest-scoring candidates (each gets a `score` and `image`
  field added).

### Scene & query generation — `app/scene_analyzer.py`, `app/query_generator.py`
Both follow the same pattern: try Claude if a key is set, else a
rule-based fallback. Both read the key live via `config.get_key(...)` /
`config.is_key_set(...)` so entering a key in the API Keys tab takes
effect on the very next run, with no restart.

**`scene_analyzer.py`**
- **`_fallback_scene_split(story)`** — splits on sentence boundaries with
  a regex; one scene per sentence.
- **`_strip_code_fence(text)`** — strips ` ```json ` fences Claude
  sometimes wraps JSON in (also imported and reused by
  `query_generator.py`).
- **`_analyze_with_claude(story)`** — one Claude call asking for a JSON
  list of scene objects (`scene_id`, `description`, `people`, `location`,
  `year`, `event`).
- **`analyze_story(story)`** — the public entry point `pipeline.py` calls:
  tries `_analyze_with_claude` if a key is set, catches any exception and
  falls back to `_fallback_scene_split`.

**`query_generator.py`**
- **`_fallback_queries(scene)`** — builds up to 4 query strings by
  combining the scene description with people/location/year/event fields.
- **`_generate_with_claude(scene)`** — one Claude call asking for 3-4
  search-query strings for the scene.
- **`generate_queries(scene)`** — the public entry point: tries Claude if
  a key is set and a non-empty list came back, else falls back.

### Image/video retrieval — `app/image_sources.py`
**`gather_candidates(query, limit_per_source, media_types, use_duckduckgo)`**
is the only function `pipeline.py` calls. It runs every source function in
turn and merges + de-duplicates (by URL) the results:

- `NO_KEY_SOURCES` (always run): `search_wikimedia`, `search_nasa`,
  `search_internet_archive`, `search_met_museum`, `search_openverse`,
  `search_loc` — each hits a different free public API and returns the
  shared candidate shape via the `_candidate()` helper.
- `OPTIONAL_KEY_SOURCES` (run, but each self-skips if its key isn't set):
  `search_flickr`, `search_pexels`, `search_pixabay` — each reads its key
  live from `config.get_key(...)`.
- `search_duckduckgo` (only if `use_duckduckgo=True`) — no-key fallback,
  with retry/backoff (`_get_ddgs_class` handles the `duckduckgo_search` →
  `ddgs` package rename) since its unofficial endpoint rate-limits hard.
- **`_candidate(url, thumbnail_url, source, title, media_type)`** — the
  one place the shared output shape is built, so every source function
  returns identically-shaped dicts.

---

## 2. Settings & keys — `app/config.py`
Used by both the main app and (indirectly, since it's the same package)
anything imported inside `worker_server.py`.

- Sets up `MODEL_DIR` / `DATA_DIR` (creating the folders if missing).
- **`KEY_SPECS`** — list of dicts describing each optional API key (id,
  display label, help text, signup URL). This is what the "API Keys" tab
  iterates over — add a new key here and it appears in the UI
  automatically.
- **`_keys`** — the actual in-memory store, seeded from environment
  variables at import time.
- **`get_key(key_id)` / `set_key(key_id, value)` / `is_key_set(key_id)`**
  — the live read/write API every consumer (`scene_analyzer.py`,
  `query_generator.py`, `image_sources.py`) should use instead of
  importing a constant, so a key entered later in the UI is picked up
  immediately.
- Module-level constants (`ANTHROPIC_API_KEY`, etc.) are kept only for
  backwards compatibility — they're a snapshot taken at import time and do
  **not** update when the UI changes a key.

---

## 3. WORKER flow — `python worker.py` (separate Colab)

### `worker.py`
The single command run in the GPU Colab notebook.

- **`install_requirements()`** — same idea as `launcher.py`: pip-installs
  `requirements.txt` (which includes `fastapi`/`uvicorn` for this side).
- **`_download_cloudflared()`** — downloads the `cloudflared` binary for
  the current CPU arch (amd64/arm64) if not already present. No account or
  token needed.
- **`start_server()`** — imports `app.worker_server.build_fastapi_app()`,
  and runs it with `uvicorn` on a background thread, bound to
  `0.0.0.0:8000` (or `$WORKER_PORT`).
- **`start_tunnel()`** — runs `cloudflared tunnel --url http://localhost:8000`
  as a subprocess, scans its stdout for the `https://*.trycloudflare.com`
  URL, and prints it prominently — that's the URL you paste into the main
  app's Workers tab.
- `if __name__ == "__main__":` → `install_requirements()` → `start_server()`
  → `start_tunnel()` → blocks on the tunnel process until you stop the
  cell.

### `app/worker_server.py` — what actually runs on the worker
**`build_fastapi_app()`** returns a FastAPI app exposing:

- **`GET /health`** — reports `{"status": "ok", "device": "cuda"|"cpu"|"unknown"}`
  by checking `torch.cuda.is_available()`. This is what
  `worker_client.check_worker()` polls to decide connected/not-connected
  and what device label to show in the UI.
- **`GET /models`** — returns `model_manager.list_models()` — the
  worker's own local disk cache, same registry/shape as the main app's.
- **`POST /models/{model_id}/download`** — starts
  `model_manager.download_model_stream()` on a background thread, storing
  live progress in the module-level `_download_jobs` dict keyed by
  `model_id`.
- **`GET /models/{model_id}/progress`** — reads that same
  `_download_jobs` dict (or reports "installed"/"not started" if no job is
  in flight) — this is what `worker_client.download_model_stream()` polls.
- **`DELETE /models/{model_id}`** — `model_manager.delete_model()`.
- **`POST /rank`** (body: `RankRequest` — text, candidates, model_id,
  top_k) — runs `clip_ranker.rank_candidates()` locally on the worker's
  GPU, then JPEG-encodes+base64s each result's thumbnail (via PIL) so it
  can travel over plain JSON back to the main app.

### `app/worker_client.py` — how the MAIN app talks to workers
Holds a registry of every worker the user has added, keyed by a short
random id. This is what `compute.py` and the Workers tab call:

- **`add_worker(url, label)`** — normalizes the URL, dedupes by URL
  (re-adding the same URL returns the existing id), registers it, returns
  the new id.
- **`remove_worker(worker_id)`** — drops it from the registry.
- **`check_worker(worker_id)`** — hits that worker's `/health`; returns
  `(connected: bool, status message)`.
- **`list_workers()`** — calls `check_worker()` for every registered
  worker and returns the full status list — this is what actually pings
  the network every time the Workers tab re-renders.
- **`connected_worker_ids()` / `is_any_connected()`** — filtered views of
  `list_workers()`, used by `compute.backend_name()`.
- **`list_models(worker_id)`**, **`download_model_stream(worker_id, model_id)`**,
  **`delete_model(worker_id, model_id)`** — thin HTTP wrappers around that
  specific worker's `/models*` endpoints; same shapes as `model_manager`'s
  local equivalents so the UI code can treat them interchangeably.
- **`rank_candidates(worker_id, text, candidates, model_id, top_k)`** —
  calls that one worker's `/rank`, then decodes each result's
  `thumbnail_base64` back into a PIL image so the return shape matches
  `clip_ranker.rank_candidates()` exactly.
- **`rank_candidates_round_robin(text, candidates, model_id, top_k)`** —
  the function `compute.py` actually calls: picks the next connected
  worker in rotation (`_rr_counter`, an `itertools.count()`), and if that
  worker's call raises, tries the next connected worker in turn before
  giving up.

---

## Where to make common changes

| I want to... | Edit this |
|---|---|
| Add a new CLIP (or other) model | `app/model_manager.py` → `MODEL_REGISTRY` |
| Add a new free/keyed image source | `app/image_sources.py` → new `search_*()` function + add it to `NO_KEY_SOURCES`/`OPTIONAL_KEY_SOURCES` |
| Add a new optional API key | `app/config.py` → `KEY_SPECS`, then read it via `config.get_key(...)` wherever it's used |
| Change how scenes are split / queries are generated | `app/scene_analyzer.py` / `app/query_generator.py` |
| Change ranking logic (e.g. a different model architecture) | `app/clip_ranker.py` (local) **and** `app/worker_server.py`'s `/rank` route (worker side) — keep both in sync |
| Change worker load-balancing (e.g. weighted instead of round-robin) | `app/worker_client.py` → `rank_candidates_round_robin()` |
| Add a new UI tab or control | `app/gradio_app.py` |
| Change what a full run does step-by-step | `app/pipeline.py` → `run_pipeline()` |
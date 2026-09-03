# Bug fixes & the new auto-model-selection flow

This documents everything changed vs. the original `main` branch. Every fix
below was verified with a runnable reproduction (either an end-to-end
pipeline run in fallback mode, or a mock HTTP worker) before and after the
change — not just read through.

## Crashes / broken code paths (would fail every run)

1. **`app/model_manager.py` — `_unload_unused_models`**
   Reassigned the module-level `_loaded_models` dict without `global`,
   which made Python treat the name as local for the *whole function* and
   raised `UnboundLocalError` on the very first line that read it. This ran
   on every VRAM-cleanup pass. Also fixed it calling `.to('cpu')` on the
   wrapper dict (`{'pipeline': ..., 'device': ...}`) instead of the actual
   model object inside it.

2. **`app/model_manager.py` — CLIP downloads silently empty**
   `smart_download_model()` passed
   `ignore_patterns=["*.safetensors", "*.bin"]` for CLIP models — excluding
   *both* possible weight formats. The folder still had `config.json`, so
   `is_installed()` reported it as installed, but loading it would fail
   with a missing-weights error. Removed the broken filter.

3. **`app/compute.py` → `app/worker_client.py` — missing functions**
   `compute.py` called `worker_client.list_models()` and
   `worker_client.rank_candidates_round_robin()`. Neither existed.
   Any model-list or CLIP-ranking call crashed with `AttributeError` the
   moment a worker was connected. Added both, plus `rank_candidates_on_worker`.

4. **`app/pipeline.py` → `app/video_models.py` — missing function**
   The "no workers connected" fallback branch called
   `video_models.generate_fallback(...)`, which didn't exist (only a
   private, differently-shaped `_generate_fallback_video` did). This meant
   the entire no-GPU-worker path — the most common case for a fresh
   setup — crashed every time. Added a public wrapper.

5. **`app/production_bible.py`, `app/script_enhancer.py` — wrong API key id**
   Both checked `config.is_key_set('anthropic')` / `get_key('anthropic')`,
   but the real key id (see `config.py`) is `'ANTHROPIC_API_KEY'`. Since the
   lookup always missed, Claude-powered bible generation and script
   enhancement silently always used the dumb rule-based fallback, even with
   a valid key configured. Also aligned the Claude model string
   (`claude-3-5-sonnet-20240620` → `claude-sonnet-4-6`) with the rest of the
   codebase, and made both response parsers join all content blocks instead
   of assuming `response.content[0]`.

6. **`app/production_bible.py`, `app/scene_orchestrator.py` — `people` type mismatch**
   `scene_analyzer.py`'s Claude path (and `query_generator.py`) treat
   `scene['people']` as a **list** of names. But `production_bible.py` and
   `scene_orchestrator.py` called `.split(',')` on it as if it were a
   comma-separated **string** — `AttributeError` any time Claude-based scene
   analysis was in use. Added `production_bible._scene_people()`, a small
   helper that accepts either shape, and used it in both places.

7. **`app/clip_ranker.py` / `app/compute.py` — CLIP model id mismatch**
   Defaulted to `"clip-vit-b-32"`, which isn't a key in
   `model_manager.MODEL_REGISTRY` (`"openai/clip-vit-base-patch32"` /
   `"openai/clip-vit-large-patch14"` are). `is_installed()` for that id
   could never be true. Introduced a shared `DEFAULT_CLIP_MODEL_ID` constant.

8. **`app/gradio_app.py`, `launcher.py` — `.launch(theme=...)`**
   `gr.Blocks.launch()` doesn't accept a `theme` kwarg (it belongs on
   `Blocks(...)`, where it was already being set) in Gradio 4/5, and raised
   `TypeError` at startup. Removed the duplicate/misplaced argument in both
   files. (Verified this also degrades gracefully — just a deprecation
   warning, no crash — under Gradio 6, and pinned `requirements.txt` to
   `gradio<6.0.0` since Gradio 6 moved `theme` to `.launch()` instead, which
   would otherwise make behavior depend on exactly when `pip install`
   resolves a fresh environment.)

9. **`app/gradio_app.py` — API Keys tab did nothing**
   The Save/Clear buttons were raw HTML `<button onclick=...>` elements
   dispatching browser `CustomEvent`s (`save_key`/`clear_key`) that nothing
   in the Python app ever listened for. Keys could only ever be set via an
   environment variable before launch, and the tab's "Save"/"Clear" buttons
   were non-functional decoration. Rebuilt with real Gradio
   `Textbox`/`Button` components wired to `config.set_key()`.

10. **`app/gradio_app.py` — gallery/video rendering**
    The generate handler pushed raw base64-encoded video strings straight
    into `gr.Gallery`/`gr.Video`, which need file paths (or URLs), not
    base64 blobs — clips would never actually render. Added
    `_b64_video_to_tempfile()` to decode each clip to a temp `.mp4` and pass
    real paths through instead.

## Missing behavior vs. the requested flow

You asked for: **script → analyze/build a bible → auto-pick the best model
for the job → install it → if a different model is needed, delete the old
one first → generate**, with heavy models confined to worker Colabs and the
main Colab only ever supervising.

The worker/heavy-model separation was already structurally correct (the
main app never calls `model_manager` directly for video — only over HTTP to
a worker), but two pieces were flat-out missing:

- **No automatic model selection existed at all.** `model_id` was just
  whatever the user picked from a fixed 3-item dropdown, sent unchanged to
  every scene. Added **`app/model_selector.py`**:
  - `select_video_model(production_units, worker_id, quality_preference)`
    picks from `stable-video-diffusion` → `zeroscope` →
    `realistic-vision` (best → lightest), using the job's scene
    count/total duration and, if a worker is connected, its *reported*
    free VRAM (`GET /health`). With no worker connected or no usable VRAM
    info, it safely defaults to the lightest model rather than assuming a
    heavy one "fits."
  - A big job (>12 scenes or >180s total) biases toward the lighter model
    even in `"auto"` mode, unless `quality_preference="quality"` is forced.
  - The Gradio dropdown now defaults to **"🧠 Auto (recommended)"**, with
    the three explicit models kept as a manual override.

- **No delete-before-install step.** Added
  `worker_client.switch_video_model(worker_id, new_model_id)`, which lists
  everything currently installed on the worker and deletes any *other*
  video model before the pipeline asks it to install the newly-selected
  one (CLIP/text models are left alone). `pipeline.py` calls this right
  after model selection and before generation starts.

- **The production bible → script enhancement → scenes chain was
  disconnected.** `script_enhancer.enhance_script()`'s output was computed
  and then discarded; `bible.scenes` stayed based on the original,
  unenhanced story. `pipeline.py` now re-runs scene analysis on the
  enhanced script (when it actually changed the text) and re-enriches with
  the bible before scene orchestration.

- **`ConsistencyManager` was built but never used** — `pipeline.py` sent an
  empty `context={}` to every worker instead of
  `ConsistencyManager.get_worker_context(unit)` (characters, locations,
  style, seeds). Now wired in, so cross-scene consistency actually reaches
  the worker.

- **The "Model Status" panel was always `{}`.** `pipeline.py` now yields a
  real `model_status` dict (`model_selector.describe_selection()`) with the
  chosen model, its size/VRAM requirement, and which worker (or "local
  fallback") is generating.

## Verification performed

- `python -m py_compile` across every file.
- Full module import pass with a stub `torch` (no GPU/heavy install
  available in the sandbox used to fix this).
- An end-to-end `run_video_pipeline()` run in no-worker/no-API-key
  (fallback) mode, asserting it reaches `'finished'` and produces a video —
  this is the exact path that used to crash on the missing
  `video_models.generate_fallback` function.
- `app/worker_client.py`'s new `list_models`, `rank_candidates_round_robin`,
  and `switch_video_model` exercised against a real (mock) HTTP server.
- `app/model_selector.py`'s VRAM-aware picking logic unit-tested across
  no-worker, big-VRAM, small-VRAM, and "auto vs. quality vs. speed" cases
  (this caught a real bug in the first draft: with no worker connected it
  was defaulting to the *heaviest* model instead of the lightest).
- `build_app()` actually constructs the Gradio `Blocks` app without error.

## Not fixed / out of scope

`structure.md`/`README.md` describe a "Stage 4: Generate Reference Images"
step (CLIP-ranked reference photos feeding img2vid) that doesn't exist in
`pipeline.py`'s actual stage list — `image_sources.py`/`clip_ranker.py`/
`compute.py` are fully implemented but currently orphaned from the main
video pipeline. Wiring that up is a real feature addition (fetching images
per scene, ranking them, passing the best one as `reference_image`), not a
bug fix, so it's flagged here rather than silently added.

# Running on Google Colab

## One-command setup (recommended)

As of `launcher.py`, you don't need to open the notebook or run separate
setup cells manually. In a fresh Colab notebook:

```python
!git clone https://github.com/Pheonix-Alpha/Universal-Documentary-Studio.git /content/uds
%cd /content/uds
!python launcher.py
```

That single cell will, in order:

1. Mount Google Drive and create `MyDrive/Universal-Documentary-Studio/`
   with `models/`, `cache/`, `projects/`, `outputs/`, `logs/`.
2. Point `HF_HOME` / `TRANSFORMERS_CACHE` / `HF_DATASETS_CACHE` /
   `TORCH_HOME` at the Drive cache folder, so models are downloaded once
   and reused on every future session — no repeated 10GB downloads.
3. Install any missing dependencies (base `requirements.txt` always;
   `torch` / `diffusers` / `transformers` / `accelerate` / `safetensors` /
   `piper-tts` / `gradio` / `ffmpeg` only if not already present).
4. Detect GPU / VRAM / RAM / CPU / disk via the existing
   `core.resource_manager.ResourceManager` (no GPU is ever assumed).
5. Ask the existing `models.registry.ModelRegistry` which models it would
   pick for the detected hardware, and download the Piper TTS voice to
   Drive if it isn't already there (real, wired-in adapter). Note: the
   image/video generation models the registry may select are pre-cached
   on Drive for future use, but `VisualAgent` in this codebase still
   always renders visuals through the mock/Ken-Burns engine — real
   diffusers-based image/video generation isn't wired in yet.
6. Write `config/config.yaml` fresh, every run, from the current Drive
   paths and detected hardware (`mock_mode` reflects whether a real TTS
   is actually ready; `local_gpu_enabled` reflects whether a GPU was
   actually detected — nothing is hard-coded).
7. Validate the install via `app.startup.run_startup()`.
8. Launch the Gradio UI (`app/ui_simple.py`: Topic / Duration / Language
   / Voice / Shorts + GENERATE, with a live progress bar and log; the
   full per-stage advanced dashboard is available in a collapsed section).

Useful flags for debugging without a full run:

```
python launcher.py --local          # skip Drive mount, use ./uds_data instead
python launcher.py --skip-deps      # skip pip installs
python launcher.py --skip-models    # skip model selection/download
python launcher.py --skip-ui        # do everything except launching Gradio
```

On your second and subsequent sessions, re-running the exact same cell
reuses everything already on Drive — no reinstalling, no re-downloading.

## Manual setup (advanced / notebook-based)

If you'd rather drive things cell-by-cell instead of via `launcher.py`:

1. Open `notebooks/colab/documentary_studio.ipynb` in Google Colab.
2. Select a GPU runtime (Runtime → Change runtime type → GPU). Any GPU is
   fine — the notebook detects VRAM at runtime and adapts; nothing is
   hard-coded to a specific card.
3. Run the setup cells. They will:
   - install dependencies
   - detect GPU / VRAM / CUDA and print a runtime report
   - initialize `ResourceManager` and `ModelRegistry`
   - **not** download any heavyweight model yet (lazy loading only)
4. Mount Google Drive (optional) if you want persistent project storage
   across sessions — recommended, since Colab sessions can terminate
   without warning.
5. Run the worker loop cell. It will accept jobs, execute GPU tasks,
   checkpoint after each one, and release model memory between jobs.

## What to expect on disconnect

Colab does not guarantee session length. If the runtime disconnects
mid-project:

- Every completed stage (research, script, scenes, per-scene assets,
  audio, render) is already checkpointed to `projects/<id>/*.json` and to
  disk (or Drive, if mounted).
- Reopen the notebook, reconnect a runtime, and re-run the worker loop
  for the same `project_id`. `DocumentaryPipeline.run_full()` resumes
  from the last completed checkpoint automatically — it does not
  restart the project.

## Safety behavior

- The notebook never assumes a fixed GPU, VRAM amount, or session length.
- `ResourceManager` always reserves a safety margin
  (`resource.vram_safety_margin_gb`, default 1.5GB) below the detected
  VRAM before considering a job "fits."
- If a requested model doesn't fit in the currently available VRAM, the
  worker automatically selects a smaller compatible model
  (`ModelRegistry.get_best_*_model`) or falls back to a non-AI technique
  (image animation, charts, licensed media) rather than crashing.

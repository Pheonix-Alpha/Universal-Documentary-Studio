# Dual-GPU Text-to-Video Pipeline

## Kaggle (generation)
1. Enable **GPU T4 x2** in Kaggle notebook settings.
2. `pip install diffusers transformers accelerate gradio imageio imageio-ffmpeg huggingface_hub`
3. Run `app_production.py`. On startup it calls `ensure_pipelines_ready()`,
   which checks each GPU's local cache and only downloads/loads what's
   actually missing — both GPUs do this in parallel threads, so a cold
   start on both isn't twice as slow as a cold start on one.
4. The Gradio app launches (public link via `share=True`) with a live
   **Status** box at the top showing both GPUs' current stage, progress,
   and last heartbeat age.
5. Outputs land in `/kaggle/working/raw_asset_gpu0.mp4` and `raw_asset_gpu1.mp4`.

### Heartbeat / status system
- Every worker (download step, VRAM load step, generation step) writes
  `{stage, progress, prompt, last_heartbeat}` per GPU to
  `/tmp/pipeline_status.json`, refreshed at least every 2 seconds via a
  dedicated heartbeat thread — independent of whether the generation
  callback itself is firing.
- If the Gradio tab disconnects and you reopen it, `app.load(..., every=2)`
  re-polls this file immediately, so you see current state, not a blank
  slate.
- A **stale** heartbeat (no update for >4 intervals) means that GPU's
  worker actually stalled or crashed — a live but slow-climbing progress
  number means it's just busy. The status text flags stale entries with
  `[STALE]`.
- You can also just `cat /tmp/pipeline_status.json` directly from a
  Kaggle terminal/notebook cell if the UI itself is unreachable.

### Idempotent model prep
- `_ensure_downloaded` tries a `local_files_only=True` cache check first;
  it only calls out to the Hub if that fails, so re-running the script
  (or the Gradio button) mid-session never re-downloads a model that's
  already local.
- `_prepare_pipeline` keeps built pipelines in a module-level dict keyed
  by GPU id; if a pipeline object already exists for that GPU, it's
  returned as-is with no re-load into VRAM.
- Both GPUs' prep runs in parallel threads in `ensure_pipelines_ready()`,
  so download/load time overlaps instead of stacking.

Notes:
- Both pipelines load the full CogVideoX-2b weights (~13-14GB fp16 each),
  so `enable_model_cpu_offload(gpu_id=N)` is required on 16GB T4s — do not
  remove it to chase "true no-offload" parallelism, or you will OOM.
- Each pipeline is still bound to its own GPU, so the two renders run
  concurrently and independently; offload just manages headroom within
  each GPU, it doesn't share state across them.

## Colab (mastering)
1. Copy the raw mp4(s) from Kaggle to Colab (Drive sync or direct download).
2. Run `bash setup_colab.sh`. It's idempotent — re-running it after a
   runtime restart skips `RealESRGAN_x4plus.pth`, the `rife_repo` clone,
   and the `train_log` copy if they're already present, instead of
   re-fetching everything from scratch.
3. Run:
   ```python
   from colab_master import process_and_upscale_stream, check_master_status
   process_and_upscale_stream("raw_asset_gpu0.mp4", "final_1080p_master.mp4")
   ```
4. If your Colab tab disconnects mid-run (common on long clips) but the
   runtime itself is still alive, reconnect and call `check_master_status()`
   in a fresh cell — it reads `/tmp/master_status.json` and reports the
   stage, `frames_done/total_frames`, and heartbeat age, so you know
   whether to keep waiting or the job actually died.

## Fixing the Real-ESRGAN install error
`pip install basicsr realesrgan` fails on current Colab/Kaggle images
with `egg_info did not run successfully` because `basicsr`'s old-style
`setup.py` needs `torch` visible inside pip's isolated build
environment, which it isn't by default. Even once that's fixed, its
code imports `torchvision.transforms.functional_tensor`, which modern
torchvision removed. `setup_colab.sh` now installs torch first, uses
`--no-build-isolation`, and `colab_master.py` shims the missing module
back in before importing `basicsr`/`realesrgan`. If you still hit
errors, re-run `setup_colab.sh` after a fresh runtime restart rather
than on top of a half-installed environment.

## Running Kaggle + any number of Colabs as one system
`distributed_queue.py` is a shared module (paste the identical file
into the Kaggle notebook and into every Colab worker notebook) that
uses a Google Drive folder as a job queue and a heartbeat board, since
Kaggle and Colab can't reach each other directly.

**One-time setup:**
1. Google Cloud Console -> create a service account -> download its JSON key.
2. Create a Drive folder, share it with the service account's email (Editor access).
3. Copy the folder ID from its URL.
4. Upload the JSON key into each notebook session (as a secret/private file, never public).
5. In `app_production.py`, set `SERVICE_ACCOUNT_JSON` and `DRIVE_ROOT_FOLDER_ID` at the top.
6. In `colab_worker.py`, set the same two values.

**Running it:**
- Start `app_production.py` on Kaggle as usual. Every finished render
  is now also pushed to the shared `pending/` queue automatically.
- Start `colab_worker.py` in one Colab notebook. It claims jobs, masters
  them, and uploads results to `done/` — no manual copying between
  platforms needed.
- Open a second (or third, fourth...) Colab notebook and run the exact
  same `colab_worker.py`. Each picks its own random worker ID and pulls
  from the same queue — more workers just means more jobs processed in
  parallel, with zero code changes.
- The Kaggle Gradio status panel merges local GPU status with every
  Colab worker's heartbeat, so one screen shows the whole system:
  which GPU is generating what, which Colab is mastering which job,
  and which nodes have gone stale.

**Leave `SERVICE_ACCOUNT_JSON = None`** in `app_production.py` to keep
running Kaggle standalone with no distributed hand-off — nothing else
changes.

**Honest limitation:** job claiming uses an optimistic rename-and-verify
check, not a real database transaction. It's fine for a handful of
cooperating workers; if two happen to poll at the exact same instant,
the loser just retries next cycle rather than double-processing. It
is not built to survive adversarial or very high-concurrency use — if
you need that guarantee, put a real transactional store (e.g.
Firestore) behind the same interface instead of raw Drive files.

## What's fixed vs. earlier drafts
- Two isolated pipeline instances, each pinned to its own GPU via
  per-instance `gpu_id` offload — no shared-pipe device collision.
- Correct `callback_on_step_end(pipe, step, timestep, kwargs) -> kwargs` signature.
- Real Real-ESRGAN weights loaded from an actual `.pth` file.
- Genuine RIFE optical-flow interpolation (8fps -> 24fps), not frame tripling,
  including the required padding to a multiple of 32 and clipping before
  the uint8 cast.
- Pillarbox padding to 16:9 with exact-width rounding before upscaling,
  so nothing gets stretched.
- Frame-by-frame `VideoWriter` streaming — no full-clip RAM buffering.
- HF cache redirected to `/tmp/hf_cache` before any DL library import.
- Heartbeat/status files on both sides for disconnect/reconnect visibility.
- Idempotent download-then-load logic on both sides, run in parallel
  across the two Kaggle GPUs at startup.

## Known residual risks worth checking before a real run
- `/tmp` on Colab/Kaggle is ephemeral session storage, not a guaranteed
  unlimited pool — check current quota, don't assume it's infinite. It's
  also wiped on a runtime restart, so `/tmp/pipeline_status.json` and
  `/tmp/master_status.json` only survive a *tab* disconnect, not an
  actual kernel/runtime recycle — those are different failure modes and
  the status file only protects against the former.
- RIFE's `Practical-RIFE` repo structure/checkpoint hosting can change;
  verify `train_log/RIFE_HDv3.py` exists after cloning before running.
- A T4 running Real-ESRGAN tiled at 4x on 1080p-target frames plus RIFE
  in the same process is memory-tight; if you hit OOM, lower `tile` size
  or process in two passes (upscale pass, then interpolation pass) instead
  of interleaving them.
- `local_files_only=True` cache checks via `snapshot_download` still walk
  the whole cache tree; on a very large repo this isn't instant, but it's
  a disk read, not a network round-trip.

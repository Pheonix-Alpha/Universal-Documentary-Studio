# Troubleshooting

## `RenderError: ffmpeg command failed`

Almost always means one of:

1. **FFmpeg isn't installed / not on PATH.** Check with `ffmpeg -version`.
2. **A zoompan/filter expression is malformed** (if you've edited
   `engines/animation/ken_burns.py`). FFmpeg's zoompan expression parser
   is picky — test any new `CameraMovement` expression against a small
   sample image/duration before wiring it into a full run.
3. **Mismatched clip parameters at concat time.** `concat_clips` first
   tries a fast stream-copy concat; if scene clips have inconsistent
   resolution/codec it automatically retries with a re-encode. If both
   fail, check that every scene clip was rendered at the same
   width/height/fps as the project config.

## Pipeline seems to skip a stage

This is very likely correct behavior, not a bug: `DocumentaryPipeline`
checks for an existing checkpoint (e.g. `script.json`) before running
that stage, and logs `"Resuming: <stage> already complete."` if found.
To force regeneration of a stage, delete its checkpoint file first:

```python
pm.store.delete_checkpoint("script.json")
```

(The Gradio dashboard's REGENERATE SCRIPT / REGENERATE VOICE / REGENERATE
SCENE buttons do this for you.)

## `ResourceError: Insufficient VRAM`

The `ResourceManager` is refusing to schedule a job because the model's
`minimum_vram_gb` exceeds the *effective* available VRAM (detected VRAM
minus the configured safety margin, default 1.5GB). This is intentional
— it prevents an OOM crash mid-job. Options:

- Lower `resource.vram_safety_margin_gb` in `config/config.yaml` (not
  recommended below ~1GB on consumer GPUs).
- Let the job fall through to a smaller model / non-AI fallback (default
  behavior — no action needed).
- Run on a worker with more VRAM (e.g. a better Colab GPU tier).

## QA score is lower than expected

Check `qa.json`'s `issues` list — each issue names its `category` and
`severity`. Common causes:

- `missing_source` / `weak_sourcing`: a claim's `source_ids` don't
  resolve to real entries in `research.sources`. Fix at the research
  stage, not by suppressing the QA check.
- `unknown_license`: an `external_media` asset has no `LicenseRecord`.
  Either drop the asset or supply real license metadata.
- `visual_reuse`: the same generated image/prompt was reused beyond the
  configured threshold (`qa/asset_checker.py`'s `max_reuse`). Consider
  varying scene prompts or regenerating the overused scene.

## Tests are slow

The full test suite runs real FFmpeg encodes (Ken Burns animation for
every `CameraMovement`, full end-to-end pipeline runs). This is
intentional — it's what makes the mock-mode tests a genuine
end-to-end guarantee rather than a mocked-out illusion. If you need a
fast subset while iterating:

```bash
PYTHONPATH=. pytest tests/ -q -k "not end_to_end and not topic_independence"
```

## `MOCK_MODE=false` but nothing downloads

By design — `app/startup.py`'s `run_startup()` never eagerly downloads a
model. Real models are loaded lazily by the scheduler on first actual
use. If you expected a specific model to load, check that
`ModelRegistry.get_best_*_model()` actually selected it (log line:
`"Selected model <name> for task=... "`) for your available VRAM and
commercial-use settings.

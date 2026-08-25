# Universal Documentary Studio (UDS)

UDS turns a single topic into **one professional documentary per day**, plus
3–5 derived Shorts, thumbnails, and metadata — with mandatory human review
before anything is considered "done."

It is **topic-agnostic**: the same pipeline handles CEO stories, historical
events, disasters, inventions, science, biographies, business
competition, and any other documentary subject, with no category
hard-coded as a special case.

## Why this exists

The developer's local machine (Ryzen 5000H, 8GB RAM, RTX 2050) cannot run
heavyweight generative AI models. UDS is architected so that:

- **Google Colab is the primary GPU worker.** Heavy jobs (AI image/video
  generation, large TTS) prefer a remote GPU and queue or fall back
  gracefully when one isn't available.
- **The local machine never runs heavy AI automatically.** Local GPU use
  requires an explicit opt-in (`local_gpu_enabled: true`).
- **Every pipeline stage is checkpointed to disk**, so a disconnected
  Colab session resumes from the last completed stage instead of
  restarting.
- **Nothing publishes itself.** A human must review and approve every
  project before it's considered final.

## Quickstart (MOCK_MODE — no GPU, no paid APIs required)

```bash
pip install -r requirements.txt
python -m app.main --topic "The Invention of the Transistor" --full
```

This produces, in `projects/<project_id>/`:

- `renders/long_form.mp4` — an 8–15 minute (target-configurable) documentary
- `shorts/short_1.mp4` … `short_N.mp4` — 3–5 independently scripted vertical Shorts
- `thumbnails/thumbnail_*.png` — 3–5 thumbnail concepts
- `qa.json`, `research.json`, `licenses/…` — QA, sourcing, and licensing reports

Launch the human-review dashboard instead:

```bash
python -m app.main --ui
```

## Running tests

```bash
pip install -r requirements.txt
PYTHONPATH=. pytest tests/ -q
```

All tests run in MOCK_MODE — no GPU, model download, or paid API is
required. See `ARCHITECTURE.md` for how MOCK_MODE substitutes real PNG /
WAV / MP4 files for expensive AI outputs so the entire pipeline is
exercised end-to-end in CI.

## Documentation

- `ARCHITECTURE.md` — system design, data flow, state machine, resource management
- `SETUP_COLAB.md` — running the GPU worker notebook
- `MODELS.md` — the model registry and how models are selected
- `LICENSES.md` — licensing policy for assets, media, and AI models
- `TROUBLESHOOTING.md` — common issues and how to diagnose them

## Development milestones

This repository was built milestone-by-milestone (see the original spec).
Every stage has real interfaces, a working mock implementation, and unit +
integration tests — nothing here is a stub or `NotImplementedError`
placeholder for core functionality.

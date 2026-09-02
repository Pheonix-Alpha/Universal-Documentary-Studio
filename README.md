# Story → Verified Image Storyboard

A prototype of the pipeline: **story → scenes → multi-query search → multi-source
image retrieval → CLIP re-ranking → storyboard**, wrapped in a single Gradio app,
built to run on a free Colab T4.

This is "Version 1–2" of the design (rule-based/LLM scene splitting, multi-query
generation, Wikimedia/Unsplash/DuckDuckGo retrieval, CLIP ranking). Version 3+
ideas (entity/date verification, VLM judging, the agentic retry loop) can be
added inside `app/pipeline.py` without touching the UI.

## Run it in Google Colab (one command after cloning)

In a Colab cell:

```python
!git clone <YOUR_REPO_URL> research_ai && cd research_ai && python launcher.py
```

That single line will:
1. `pip install` everything in `requirements.txt` (progress printed to the cell)
2. Launch the Gradio app with `share=True`, printing a public URL you can open
   (Colab doesn't expose `localhost` directly, so use that link, not `127.0.0.1`)

**Optional, for better results** — run this in a cell *before* the command above:

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."      # enables LLM scene/query analysis
os.environ["UNSPLASH_ACCESS_KEY"] = "..."           # adds Unsplash as an image source
```

Without these, the app still works: scenes are split with a rule-based sentence
splitter, queries are generated with simple templates, and images come from
Wikimedia Commons + DuckDuckGo image search (no keys required for either).

## Using the app

- **Generate Storyboard tab**: paste a story, pick a CLIP model, hit **Start**.
  You'll see the current stage, a live percentage bar, a scrolling log, and the
  ranked images streaming into the gallery scene by scene.
- **Models tab**: every model in the registry is listed with its size and
  install status. Not downloaded → **Download** button with its own percentage
  bar. Installed → **Delete** button to free disk space (useful since Colab's
  disk is limited and shared with everything else in the runtime).

The CLIP model dropdown on the Generate tab won't work until you've downloaded
that model in the Models tab — the app will tell you if you try to run before
downloading one.

## Project layout

```
research_ai/
├── launcher.py          # single entry point: pip install -> launch Gradio
├── requirements.txt
├── app/
│   ├── config.py         # paths + optional API keys from env vars
│   ├── model_manager.py  # model registry, download-with-progress, delete
│   ├── scene_analyzer.py # story -> scenes (Claude API or rule-based fallback)
│   ├── query_generator.py# scene -> 3-4 search queries
│   ├── image_sources.py  # Wikimedia Commons / Unsplash / DuckDuckGo search
│   ├── clip_ranker.py    # CLIP text<->image similarity ranking
│   ├── pipeline.py       # orchestrates the stages, yields progress for the UI
│   └── gradio_app.py     # the UI itself
├── models/               # downloaded model weights (gitignored)
└── data/                 # local cache (gitignored)
```

## Extending toward the full design

- **Entity / date / location verification** (Version 3): add a scoring function
  in `pipeline.py` that compares each candidate's metadata (where available)
  against the scene JSON, and blend it into the CLIP score using the weighted
  formula from the design doc (semantic + entity + location + time + source).
- **VLM verification** (Version 4): add a `vlm_verifier.py` that asks a
  vision-language model "does this image actually depict `<scene>`?" and use
  it to filter/re-rank the CLIP shortlist before display.
- **Agent loop** (Version 5): wrap the per-scene retrieval in `pipeline.py` in
  a loop that re-generates queries and searches again when the best score is
  below a confidence threshold, instead of accepting the first pass.
- **Evaluation** (§14 of the design): build a small labeled test set (correct /
  acceptable / incorrect image per scene) and compute Recall@k / Precision@5 /
  MRR to actually measure whether changes help.

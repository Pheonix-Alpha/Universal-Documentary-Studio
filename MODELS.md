# Models

UDS is **provider-agnostic and free/open-source-first**. No paid API is
required for the core pipeline to function.

## How model selection works

Every model registers a `ModelCapability` (see `models/capabilities.py`)
declaring:

- `model_name`, `provider`, `version`, `license`
- `minimum_vram_gb`, `recommended_vram_gb`, `estimated_disk_gb`
- `supported_resolution`, `supported_tasks`
- `commercial_use` (boolean)
- `status` (`available` / `disabled` / `experimental`)
- `quality_rank` (used to break ties among compatible models)

`ModelRegistry.get_best_model(task, available_vram, commercial_use)`
returns the **highest quality_rank model that fits** the available VRAM
and commercial-use requirement. If nothing fits, it returns `None` and
the calling agent must fall back to a non-AI technique — the registry
never crashes or silently downgrades output without the caller knowing.

## Default catalog

| Model | Task | Provider | Min VRAM | License | Commercial use |
|---|---|---|---|---|---|
| mock-image-v1 | image_generation | mock | 0 GB | internal-testing | yes |
| sd-turbo-small | image_generation | stability-open | 4 GB | OpenRAIL-M | yes |
| sdxl-base | image_generation | stability-open | 8 GB | OpenRAIL-M | yes |
| mock-video-v1 | video_generation | mock | 0 GB | internal-testing | yes |
| svd-open | video_generation | stability-open | 12 GB | OpenRAIL-M | yes |
| mock-tts-v1 | tts | mock | 0 GB | internal-testing | yes |
| piper-tts | tts | rhasspy-piper | 0 GB | MIT | yes |
| coqui-xtts | tts | coqui | 4 GB | Coqui Public Model License | **no** |

`coqui-xtts` is registered but flagged `commercial_use=False` — it will
never be selected when a caller requests `commercial_use=True` (the
default throughout the pipeline). This is intentional: **the registry
never assumes an open-source model automatically permits commercial
use.**

Add more entries via `config/models.yaml` (documentation/reference) and
`ModelRegistry.register()` (actual registration) without touching any
calling code.

## MOCK_MODE models

`mock-image-v1`, `mock-video-v1`, and `mock-tts-v1` are always available
(0 GB VRAM requirement) and are what the test suite and MOCK_MODE runs
use by default. They produce real files (PNG/MP4/WAV) via Pillow/FFmpeg
rather than stub placeholders, so the entire downstream pipeline
(animation, mixing, rendering, QA) is exercised the same way it would be
with a real model swapped in.

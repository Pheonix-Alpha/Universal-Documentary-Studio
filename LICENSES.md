# Licensing Policy

## Assets

Every asset (`core/models.py::Asset`) tracks:

- `origin`: `ai_generated` / `human_created` / `external_media`
- `license`: a `LicenseRecord` (license string, commercial-use flag,
  attribution requirement/text) — **required** for `external_media`
  assets

`qa/license_checker.py` rejects (marks as a critical QA issue) any
`external_media` asset whose license is missing or `"unknown"`. This is
enforced automatically as part of every QA run — it is not optional.

`adapters/media_sources/mock_provider.py` deliberately returns **zero
results** rather than fabricating license metadata for any media it
can't actually verify. A real media-source adapter (Wikimedia Commons,
NASA archive APIs, etc.) must supply real, verifiable license data for
every result or return nothing.

## AI-generated content

Every AI-generated asset is tagged `origin=ai_generated` and tracked in
the asset manifest (`ai_assets.json`-equivalent is the `AssetManifest`
checkpoint). Generated historical scenes are never presented as genuine
archival footage — they are stock-in-trade documentary re-creations,
which is standard and disclosed practice.

## Models

See `MODELS.md`. Every model in the registry declares its own license
and a `commercial_use` flag. A non-commercial-licensed model (e.g.
`coqui-xtts`) is automatically excluded from selection whenever a caller
requests `commercial_use=True`, which is the default everywhere in the
pipeline.

## Music / SFX

`engines/music/music_engine.py` and `engines/sfx/sfx_engine.py` look up
licensed/free tracks from `assets/music/<mood>/` and `assets/sfx/<category>/`
first. If no library asset exists for a given mood/category (e.g. a
fresh checkout), they synthesize a short placeholder tone via FFmpeg
rather than silently using an unlicensed file from elsewhere on disk.

## Source attribution (research, not media)

Every `Claim` in `research.json` must reference at least one `Source`
by `source_id`. `qa/source_checker.py` flags any claim with no valid
source reference as a critical QA issue, and low-confidence claims
backed by only one source as a warning.

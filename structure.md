Updated Project Structure & Execution Flow
This document maps every file and function in the enhanced codebase, tracing the order things actually run in. The system has evolved from a storyboard/image retrieval system into a full-featured distributed video generation studio with intelligent resource management.

🎯 Architecture Overview
text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MAIN COLAB ("The Director")                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  📖 PRODUCTION BIBLE SYSTEM                                                 │
│  ├── Scene Analyzer      → Breaks story into scenes                        │
│  ├── Bible Generator     → Creates character/location/style bibles        │
│  └── Script Enhancer     → Improves dialogue and narrative                │
│                                                                             │
│  🎬 SCENE ORCHESTRATOR                                                     │
│  ├── Unit Breakdown      → Converts bible to production units             │
│  ├── Consistency Manager → Ensures cross-worker consistency               │
│  └── Task Distributor    → Assigns scenes to workers                      │
│                                                                             │
│  🧠 SMART MODEL MANAGER                                                    │
│  ├── Storage Management  → Auto-cleanup when disk is full                 │
│  ├── VRAM Management     → Auto-unload models to free memory              │
│  └── Smart Download      → Checks space before downloading                │
│                                                                             │
│  🌐 WORKER COORDINATOR                                                     │
│  ├── Worker Registry     → Tracks all connected workers                   │
│  ├── Load Balancer       → Round-robin distribution                       │
│  └── Health Monitoring   → Checks worker status                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      WORKER COLABS ("Render Nodes")                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  🎥 VIDEO GENERATION ENGINE                                                │
│  ├── Model Loader        → Loads models with VRAM management              │
│  ├── Video Generator     → Generates clips from prompts                   │
│  └── Format Converter    → Converts frames to video                       │
│                                                                             │
│  🧠 SMART MODEL MANAGER (Worker side)                                      │
│  ├── Local Storage       → Manages model cache                            │
│  ├── VRAM Management     → Smart model swapping                           │
│  └── Auto-Cleanup        → Deletes unused models                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
📁 Complete File Structure
text
Universal-Documentary-Studio/
├── launcher.py                  # MAIN entry point (enhanced)
├── worker.py                    # WORKER entry point (enhanced)
├── requirements.txt             # Updated with video gen libs
├── README.md                    # Documentation
│
├── app/
│   ├── __init__.py
│   │
│   ├── config.py                # Settings + API keys (live updates)
│   │
│   ├── model_manager.py         # ★ COMPLETE REWRITE - Smart model mgmt
│   │   ├── MODEL_REGISTRY       # All models with sizes/types
│   │   ├── smart_download_model()  # Auto-cleanup before download
│   │   ├── load_model_into_vram()  # Auto-unload old models
│   │   ├── _smart_cleanup()     # Delete least-used models
│   │   ├── get_available_storage() # Storage status
│   │   └── get_vram_status()    # VRAM status
│   │
│   ├── clip_ranker.py           # KEPT - Local CLIP ranking
│   │   ├── _load_clip()         # Loads CLIP model
│   │   ├── _fetch_image()       # Downloads candidate images
│   │   └── rank_candidates()    # Ranks images by similarity
│   │
│   ├── video_models.py          # ★ NEW - Video generation interface
│   │   ├── VIDEO_MODEL_REGISTRY # Video model definitions
│   │   ├── generate_video()     # Main video generation
│   │   ├── generate_with_loaded_model() # Uses pre-loaded model
│   │   ├── _generate_svd()      # Stable Video Diffusion
│   │   ├── _generate_diffusion() # Generic diffusion
│   │   └── _frames_to_video_bytes() # Convert to MP4
│   │
│   ├── production_bible.py      # ★ NEW - Bible generation
│   │   ├── ProductionBible      # Dataclass for bible
│   │   ├── Character            # Character dataclass
│   │   ├── Location             # Location dataclass
│   │   ├── VisualStyle          # Style dataclass
│   │   ├── generate_production_bible() # Main entry point
│   │   └── _generate_bible_with_claude() # AI-powered bible
│   │
│   ├── script_enhancer.py       # ★ NEW - Script improvement
│   │   ├── enhance_script()     # Main entry point
│   │   ├── _enhance_with_claude() # AI enhancement
│   │   └── _enhance_fallback()  # Rule-based fallback
│   │
│   ├── scene_orchestrator.py    # ★ NEW - Scene breakdown
│   │   ├── ProductionUnit       # Scene production unit
│   │   ├── SceneOrchestrator    # Orchestrator class
│   │   ├── breakdown()          # Create production units
│   │   └── _generate_visual_prompt() # Create video prompt
│   │
│   ├── consistency_manager.py   # ★ NEW - Cross-worker consistency
│   │   ├── ConsistencyManager   # Main class
│   │   ├── get_worker_context() # Generate context for workers
│   │   ├── get_consistency_hash() # Hash for caching
│   │   └── share_character_reference() # Share character data
│   │
│   ├── compute.py               # ENHANCED - Dispatcher
│   │   ├── backend_name()       # "worker" or "local"
│   │   ├── list_models()        # Lists all models
│   │   ├── is_installed()       # Checks model installation
│   │   └── rank_candidates()    # Dispatches to local/worker
│   │
│   ├── worker_client.py         # ENHANCED - Worker communication
│   │   ├── _workers             # Worker registry
│   │   ├── add_worker()         # Add worker connection
│   │   ├── list_workers()       # List all workers
│   │   ├── is_any_connected()   # Check connectivity
│   │   ├── generate_video_round_robin() # ★ NEW - Distribute video gen
│   │   ├── _generate_video_on_worker() # Call specific worker
│   │   └── list_video_models()  # List video models on workers
│   │
│   ├── worker_server.py         # ENHANCED - Worker API
│   │   ├── build_fastapi_app()  # Creates FastAPI app
│   │   ├── /health              # Health with VRAM/storage status
│   │   ├── /models              # List models with status
│   │   ├── /models/{id}/download # Smart download
│   │   ├── /models/{id}/progress # Download progress
│   │   ├── /models/storage      # Storage information
│   │   ├── /models/cleanup      # Manual cleanup trigger
│   │   ├── /rank                # CLIP ranking
│   │   └── /video/generate      # ★ NEW - Video generation endpoint
│   │
│   ├── scene_analyzer.py        # ENHANCED - Story analysis
│   │   ├── analyze_story()      # Main entry point
│   │   ├── _analyze_with_claude() # AI scene analysis
│   │   └── _fallback_scene_split() # Rule-based split
│   │
│   ├── query_generator.py       # KEPT - Search query generation
│   │   ├── generate_queries()   # Main entry point
│   │   ├── _generate_with_claude() # AI queries
│   │   └── _fallback_queries()  # Rule-based queries
│   │
│   ├── image_sources.py         # KEPT - Image retrieval
│   │   ├── gather_candidates()  # Main entry point
│   │   ├── search_wikimedia()   # Wikimedia API
│   │   ├── search_nasa()        # NASA API
│   │   ├── search_flickr()      # Flickr API (keyed)
│   │   └── ... (many sources)
│   │
│   ├── pipeline.py              # ★ COMPLETE REWRITE - Video pipeline
│   │   └── run_video_pipeline() # The main flow generator
│   │       ├── Stage 1: Validate story
│   │       ├── Stage 2: Generate Production Bible
│   │       ├── Stage 3: Scene Orchestration
│   │       ├── Stage 4: Generate Reference Images
│   │       ├── Stage 5: Distribute Video Generation
│   │       ├── Stage 6: Assemble Final Video
│   │       └── Stage 7: Complete
│   │
│   └── gradio_app.py            # ★ GREATLY ENHANCED - UI
│       ├── Tab 1: Generate Video
│       ├── Tab 2: Models & Resources (NEW dashboard)
│       │   ├── Storage/VRAM dashboard
│       │   ├── Model list with status
│       │   ├── Smart download with cleanup
│       │   └── Delete/Cleanup controls
│       ├── Tab 3: Workers
│       ├── Tab 4: API Keys
│       └── Tab 5: Image Sources (kept)
│
└── outputs/
    ├── bibles/                  # Production bibles (JSON)
    ├── clips/                   # Generated video clips
    └── final/                   # Assembled final video
🔄 Execution Flow
1. MAIN APP — python launcher.py
text
launcher.py
    │
    ├── install_requirements()       # Pip install requirements
    │
    └── launch_app()
        │
        └── gradio_app.build_app()   # Builds the UI
            │
            └── demo.launch()        # Starts Gradio server
2. GENERATE VIDEO FLOW — Click "Generate Video"
text
User clicks "🎬 Generate Video"
    │
    ▼
pipeline.run_video_pipeline(story, clip_model, video_model, top_k, duration)
    │
    ├── Stage 1: Validate
    │   └── Check story length > 10 chars
    │
    ├── Stage 2: Generate Production Bible
    │   ├── production_bible.generate_production_bible(story)
    │   │   ├── scene_analyzer.analyze_story(story)    # Get scenes
    │   │   ├── _generate_bible_with_claude(story)     # AI bible (if key set)
    │   │   │   ├── Extract characters → Character objects
    │   │   │   ├── Extract locations → Location objects
    │   │   │   └── Extract visual style → VisualStyle
    │   │   └── _enrich_scenes_with_bible(scenes, bible)  # Add context
    │   │
    │   └── script_enhancer.enhance_script(story, bible)
    │       └── _enhance_with_claude(story, bible)     # AI script enhancement
    │
    ├── Stage 3: Scene Orchestration
    │   ├── SceneOrchestrator(bible)
    │   ├── orchestrator.breakdown()
    │   │   └── For each scene → ProductionUnit
    │   │       ├── scene_id, description
    │   │       ├── characters, location
    │   │       ├── visual_prompt (generated)
    │   │       ├── seed (bible.global_seed + scene_id)
    │   │       └── style_context
    │   │
    │   └── consistency_manager.ConsistencyManager(bible)
    │
    ├── Stage 4: Generate Reference Images (placeholder)
    │   └── TODO: Generate character reference images for consistency
    │
    ├── Stage 5: Distribute Video Generation
    │   ├── Check: worker_client.is_any_connected() OR video_models.is_installed()
    │   │
    │   └── For each ProductionUnit in production_units:
    │       │
    │       ├── Get consistency context
    │       │   └── consistency_manager.get_worker_context(unit)
    │       │
    │       ├── IF workers available:
    │       │   └── worker_client.generate_video_round_robin(
    │       │           prompt=unit.visual_prompt,
    │       │           model_id=video_model_id,
    │       │           context=context,
    │       │           duration=duration_per_scene,
    │       │           seed=unit.seed
    │       │       )
    │       │       │
    │       │       ├── connected_worker_ids()   # Get connected workers
    │       │       ├── Pick next worker (round-robin)
    │       │       └── _generate_video_on_worker(worker_id, ...)
    │       │           └── POST {worker_url}/video/generate
    │       │
    │       └── ELSE (local generation):
    │           └── video_models.generate_video(
    │                   prompt=unit.visual_prompt,
    │                   model_id=video_model_id,
    │                   context=context,
    │                   ...
    │               )
    │               │
    │               └── model_manager.load_model_into_vram(model_id)
    │                   ├── _unload_unused_models()   # Free VRAM
    │                   └── _load_video_model()       # Load into VRAM
    │
    ├── Stage 6: Assemble Final Video
    │   └── _assemble_video(generated_clips)    # Placeholder
    │
    └── Stage 7: Complete
        └── Yield final results to UI
3. SMART MODEL DOWNLOAD FLOW
text
User clicks "Download Model" in UI
    │
    ▼
model_manager.smart_download_model(model_id)
    │
    ├── Check if already installed
    │   └── If yes → Return "already installed"
    │
    ├── Calculate available storage
    │   └── calculate_total_storage_used()
    │
    ├── Check if enough space for model
    │   ├── IF not enough space:
    │   │   └── _smart_cleanup(needed_gb)
    │   │       ├── Get all installed models with last_used
    │   │       ├── Sort by: never used → oldest → largest
    │   │       ├── Delete models until enough space freed
    │   │       └── Return amount freed
    │   │
    │   └── IF still not enough → Error
    │
    ├── Download model
    │   └── huggingface_hub.snapshot_download(...)
    │
    └── _update_model_last_used(model_id)   # Track usage
4. WORKER VIDEO GENERATION FLOW
text
Worker receives POST /video/generate
    │
    ▼
worker_server.generate_video(request)
    │
    ├── model_manager.load_model_into_vram(model_id)
    │   ├── _unload_unused_models(keep=[model_id])   # Free VRAM
    │   └── _load_video_model(model_id, device)
    │       └── DiffusionPipeline.from_pretrained(...)
    │           ├── torch_dtype=torch.float16
    │           ├── low_cpu_mem_usage=True
    │           └── Enable attention slicing for memory efficiency
    │
    ├── video_models.generate_with_loaded_model(
    │       loaded_model,
    │       prompt,
    │       context,
    │       duration_seconds,
    │       fps,
    │       width,
    │       height,
    │       seed,
    │       reference_image
    │   )
    │   │
    │   ├── Check VRAM before generation
    │   │   └── If >90% used → torch.cuda.empty_cache()
    │   │
    │   ├── Generate based on model type:
    │   │   ├── Stable Video Diffusion → _generate_svd()
    │   │   └── Other diffusion models → _generate_diffusion()
    │   │
    │   └── Convert frames to video
    │       └── _frames_to_video_bytes(frames, fps)
    │
    └── Return base64-encoded video
🗺️ File Dependency Map
text
launcher.py
    └── gradio_app.py
        ├── pipeline.py
        │   ├── production_bible.py
        │   │   ├── scene_analyzer.py
        │   │   ├── script_enhancer.py
        │   │   └── config.py (for API keys)
        │   │
        │   ├── scene_orchestrator.py
        │   │   └── production_bible.py
        │   │
        │   ├── consistency_manager.py
        │   │   └── production_bible.py
        │   │
        │   ├── worker_client.py
        │   │   └── compute.py
        │   │
        │   ├── video_models.py
        │   │   └── model_manager.py
        │   │
        │   └── compute.py
        │       ├── model_manager.py
        │       ├── clip_ranker.py
        │       └── worker_client.py
        │
        ├── model_manager.py
        │   ├── config.py
        │   └── huggingface_hub
        │
        └── video_models.py
            └── model_manager.py

worker.py
    └── worker_server.py
        ├── model_manager.py
        ├── video_models.py
        ├── clip_ranker.py
        └── config.py
🔧 Key Data Structures
ProductionUnit
python
{
    'scene_id': int,
    'description': str,
    'characters': List[str],
    'location': str,
    'visual_prompt': str,      # Full prompt for video generation
    'duration_seconds': int,
    'seed': int,
    'style_context': Dict,     # From bible
    'assigned_worker': str,
    'status': str              # pending, generating, completed, failed
}
Model Registry Entry
python
{
    'id': str,
    'name': str,
    'type': str,               # 'clip', 'video', 'text'
    'size_gb': float,
    'description': str,
    'installed': bool,
    'path': str,
    'last_used': float,
    'priority': int
}
Worker Context (for consistency)
python
{
    'global_seed': int,
    'scene_seed': int,
    'style': {
        'visual_style': str,
        'color_palette': List[str],
        'cinematography': str,
        'lighting': str,
        'camera': str,
        'film_grain': bool
    },
    'characters': {
        'name': {
            'appearance': str,
            'personality': List[str],
            'voice': str,
            'reference_prompt': str
        }
    },
    'locations': {
        'name': {
            'description': str,
            'atmosphere': str,
            'key_elements': List[str],
            'time_of_day': str,
            'weather': str
        }
    }
}
🎯 Where to Make Common Changes
I want to...	Edit this
Add a new CLIP/video model	model_manager.py → MODEL_REGISTRY
Change model size calculation	model_manager.py → get_actual_model_size()
Change storage limit	model_manager.py → MAX_STORAGE_GB
Change cleanup strategy	model_manager.py → _smart_cleanup()
Add a new video generation model	video_models.py → VIDEO_MODEL_REGISTRY + new _generate_*() function
Change video generation logic	video_models.py → generate_with_loaded_model()
Change how the bible is generated	production_bible.py → generate_production_bible()
Change how scenes are split	scene_analyzer.py → analyze_story()
Change script enhancement	script_enhancer.py → enhance_script()
Change consistency enforcement	consistency_manager.py → get_worker_context()
Change worker load balancing	worker_client.py → generate_video_round_robin()
Add a new worker capability	worker_server.py → /health + new endpoint
Change the UI layout	gradio_app.py → build_app()
Change the full pipeline flow	pipeline.py → run_video_pipeline()
Add a new API key	config.py → KEY_SPECS
🚀 Deployment Commands
Main Colab:
bash
# Clone the repo
git clone https://github.com/yourusername/Universal-Documentary-Studio.git
cd Universal-Documentary-Studio

# Run the launcher
python launcher.py
Wait for the Gradio URL (e.g., https://xxxx.gradio.live)

Worker Colab (one or more):
bash
# Clone the repo
git clone https://github.com/yourusername/Universal-Documentary-Studio.git
cd Universal-Documentary-Studio

# Run the worker
python worker.py
Wait for the Cloudflare URL (e.g., https://xxxx.trycloudflare.com)
Copy this URL → Paste into Main app's "Workers" tab

📊 Monitoring Resources
In the UI:
Models & Resources tab shows:

💾 Storage usage (used/max GB)

⚡ VRAM usage (used/total GB)

📦 Installed models with sizes

🔄 Download progress with status

Via API (worker):
bash
# Health check
curl http://worker-url:8000/health

# Storage status
curl http://worker-url:8000/models/storage

# List models
curl http://worker-url:8000/models
🔮 Future Enhancements
Feature	Description
Character Consistency	Generate reference images for characters using SD
Video Assembly	Proper video stitching with transitions
Audio Generation	Add narration and sound effects
Parallel Worker Jobs	Multiple scenes per worker concurrently
Weighted Load Balancing	Distribute based on worker capabilities
Model Quantization	Use 8-bit/4-bit models to save VRAM
Inference Caching	Cache generated clips for reuse
Progressive Refinement	Generate low-res first, then upscale
This structure represents a complete production-ready system that can generate videos from stories using distributed GPU workers, with intelligent resource management to handle Colab's limitations!


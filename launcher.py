#!/usr/bin/env python3
"""
Universal Documentary Studio - Main Launcher
Run this in your primary Colab notebook.
"""

import subprocess
import sys
import os


def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Requirements installed.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False


def launch_app():
    """Launch the main application"""
    # Add app directory to path
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    
    from app.gradio_app import build_app
    import gradio as gr
    
    print("🚀 Launching Universal Documentary Studio...")
    demo = build_app()
    # gr.Blocks.launch() has no `theme` kwarg -- the theme is set once, at
    # Blocks(...) construction time inside build_app(). Passing it here
    # raised "TypeError: launch() got an unexpected keyword argument
    # 'theme'" and crashed on every startup.
    demo.queue(max_size=20).launch(share=True, debug=False)


def main():
    print("""
    ╔═══════════════════════════════════════════╗
    ║   🎬 Universal Documentary Studio        ║
    ║   Powered by AI Video Generation         ║
    ╚═══════════════════════════════════════════╝
    """)
    
    if not install_requirements():
        sys.exit(1)
    
    launch_app()


if __name__ == "__main__":
    main()
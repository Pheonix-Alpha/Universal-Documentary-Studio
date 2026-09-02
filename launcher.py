#!/usr/bin/env python3
"""
Single-command launcher for the Story -> Verified Image Storyboard project.

Usage (after `git clone` this repo, from inside its folder):

    python launcher.py

This will:
  1. pip-install everything in requirements.txt (prints progress)
  2. create local `models/` and `data/` cache folders
  3. launch the Gradio web app with a public share link (handy in Colab,
     since Colab doesn't expose localhost directly)

Optional environment variables (set them in a cell BEFORE running this,
e.g. `import os; os.environ["ANTHROPIC_API_KEY"] = "..."`):

  ANTHROPIC_API_KEY   -> enables LLM-based scene/query analysis (recommended)
  UNSPLASH_ACCESS_KEY -> enables Unsplash as an extra image source

Without these, the app still runs using rule-based scene splitting and
Wikimedia Commons + DuckDuckGo image search only.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS_FILE = os.path.join(ROOT, "requirements.txt")


def install_requirements():
    print("=" * 70)
    print("STEP 1/2 -- Installing dependencies (this can take a minute)")
    print("=" * 70)
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--disable-pip-version-check", "-r", REQUIREMENTS_FILE,
    ]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print("\nDependency installation failed -- see the log above.")
        sys.exit(1)
    print("\nAll dependencies installed.\n")


def launch_app():
    print("=" * 70)
    print("STEP 2/2 -- Launching Gradio app")
    print("=" * 70)
    sys.path.insert(0, ROOT)
    from app.gradio_app import build_app

    demo = build_app()
    # share=True gives you a public URL, which is what you need on Colab
    demo.queue().launch(share=True, debug=False)


if __name__ == "__main__":
    install_requirements()
    launch_app()

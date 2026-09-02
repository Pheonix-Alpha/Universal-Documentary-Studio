#!/usr/bin/env python3
"""
Single-command launcher for the GPU WORKER side of this project.

Run this in a SEPARATE Colab notebook from the one running launcher.py
(Runtime -> Change runtime type -> pick a GPU for this one):

    !git clone <this repo url>
    %cd Universal-Documentary-Studio
    !python worker.py

This will:
  1. pip-install everything in requirements.txt (prints progress)
  2. start the worker's FastAPI app (app/worker_server.py) on localhost
  3. open a public tunnel to it with cloudflared (no account/signup needed)
  4. print a public URL

Copy that URL into the "Worker" tab of the MAIN app (the one launched by
`python launcher.py` in your other notebook) and click Connect. From then
on, model downloads/deletes and CLIP ranking in the main app automatically
run here instead of on the main notebook's own CPU/GPU.

Just stop this cell to shut the worker down.
"""
import os
import platform
import re
import stat
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS_FILE = os.path.join(ROOT, "requirements.txt")
CLOUDFLARED_PATH = os.path.join(ROOT, "cloudflared")
PORT = int(os.environ.get("WORKER_PORT", "8000"))


def install_requirements():
    print("=" * 70)
    print("STEP 1/3 -- Installing dependencies (this can take a minute)")
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


def _download_cloudflared():
    if os.path.exists(CLOUDFLARED_PATH):
        return
    machine = platform.machine().lower()
    arch = "arm64" if ("arm" in machine or "aarch64" in machine) else "amd64"
    url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{arch}"
    print(f"Downloading cloudflared ({arch})...")
    urllib.request.urlretrieve(url, CLOUDFLARED_PATH)
    os.chmod(CLOUDFLARED_PATH, os.stat(CLOUDFLARED_PATH).st_mode | stat.S_IEXEC)


def start_server():
    print("=" * 70)
    print("STEP 2/3 -- Starting worker API")
    print("=" * 70)
    sys.path.insert(0, ROOT)
    import uvicorn

    from app.worker_server import build_fastapi_app

    fastapi_app = build_fastapi_app()

    def _run():
        uvicorn.run(fastapi_app, host="0.0.0.0", port=PORT, log_level="warning")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    time.sleep(2)
    print(f"Worker API is running locally on port {PORT}.\n")


def start_tunnel():
    print("=" * 70)
    print("STEP 3/3 -- Opening a public tunnel (cloudflared)")
    print("=" * 70)
    _download_cloudflared()
    proc = subprocess.Popen(
        [CLOUDFLARED_PATH, "tunnel", "--url", f"http://localhost:{PORT}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    url = None
    for line in proc.stdout:
        print(line, end="")
        match = re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", line)
        if match and url is None:
            url = match.group(0)
            print("\n" + "=" * 70)
            print("WORKER READY")
            print(f"Paste this URL into the main app's Worker tab and click Connect:\n\n    {url}\n")
            print("=" * 70 + "\n")
    return proc


if __name__ == "__main__":
    install_requirements()
    start_server()
    tunnel_proc = start_tunnel()
    try:
        tunnel_proc.wait()
    except KeyboardInterrupt:
        tunnel_proc.terminate()
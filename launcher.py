#!/usr/bin/env python3
"""
Universal Documentary Studio - Main Launcher
Run this in your primary Colab notebook.
"""

import subprocess
import sys
import os
from app import cache_manager
from app import runtime_manager
from app import capability_manager


def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
        print("✅ Requirements installed.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False


def mount_google_drive():
    """Mount Google Drive when running inside Google Colab."""

    try:
        from google.colab import drive

        drive_path = "/content/drive"

        # Already mounted
        if os.path.exists(os.path.join(drive_path, "MyDrive")):
            print("✅ Google Drive already mounted")
            return True

        print("📁 Google Drive is not mounted.")
        print("ℹ️ Please run this in a separate Colab cell:")
        print()
        print("   from google.colab import drive")
        print('   drive.mount("/content/drive")')
        print()

        return False

    except ImportError:
        print("ℹ️ Not running inside Google Colab. Drive cache disabled.")
        return False

    except Exception as e:
        print(f"⚠️ Google Drive check failed: {e}")
        return False


def launch_main_api():
    """Start the Main FastAPI server in a background thread."""
    import threading
    import uvicorn

    from app.main_server import build_main_app

    app = build_main_app()

    def run_server():
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
        )

    thread = threading.Thread(
        target=run_server,
        daemon=True,
    )
    thread.start()

    print("🌐 Main API started on port 8000")


def start_main_tunnel(port=8000):
    """Expose the Main API through a temporary Cloudflare tunnel."""
    import platform
    import subprocess
    import re

    binary_name = "cloudflared.exe" if platform.system() == "Windows" else "cloudflared"
    binary_path = os.path.join(os.getcwd(), binary_name)

    if not os.path.exists(binary_path):
        print("🌐 cloudflared not found. Downloading...")

        system = platform.system()
        machine = platform.machine()

        if system == "Linux":
            if machine in ("x86_64", "amd64"):
                url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
            elif machine in ("aarch64", "arm64"):
                url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64"
            else:
                raise RuntimeError(f"Unsupported architecture: {machine}")
        elif system == "Darwin":
            if machine in ("x86_64", "amd64"):
                url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64"
            elif machine in ("aarch64", "arm64"):
                url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-arm64"
            else:
                raise RuntimeError(f"Unsupported architecture: {machine}")
        elif system == "Windows":
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
        else:
            raise RuntimeError(f"Unsupported OS: {system}")

        import urllib.request

        urllib.request.urlretrieve(url, binary_path)

        if system != "Windows":
            import stat

            st = os.stat(binary_path)
            os.chmod(binary_path, st.st_mode | stat.S_IEXEC)

    print("🔗 Starting Main API Cloudflare tunnel...")

    cmd = [
        binary_path,
        "tunnel",
        "--url",
        f"http://localhost:{port}",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    main_api_url = None

    for line in iter(process.stdout.readline, ""):
        print(line.strip())

        if "https://" in line and ".trycloudflare.com" in line:
            match = re.search(
                r"https://[a-zA-Z0-9-]+\.trycloudflare\.com",
                line,
            )

            if match:
                main_api_url = match.group(0)

                print()
                print("=" * 60)
                print(f"🌐 MAIN API URL: {main_api_url}")
                print("=" * 60)
                print()

                break

    return main_api_url, process


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
    mount_google_drive()
    cache_manager.initialize_cache()
    runtime_manager.print_runtime_report(role="main")
    capability_manager.print_capability_report()
    launch_main_api()


main_api_url, main_tunnel_process = start_main_tunnel(port=8000)

if not main_api_url:
    print("❌ Failed to create Main API tunnel.")
    sys.exit(1)

launch_app()


if __name__ == "__main__":
    main()

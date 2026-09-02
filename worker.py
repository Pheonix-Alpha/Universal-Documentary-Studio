#!/usr/bin/env python3
"""
Universal Documentary Studio - GPU Worker
Run this in a separate Colab notebook with GPU runtime.
"""

import subprocess
import sys
import os
import platform
import time
import threading
import urllib.request
import stat


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


def download_cloudflared():
    """Download cloudflared binary"""
    print("🌐 Downloading cloudflared...")
    
    system = platform.system()
    machine = platform.machine()
    
    if system == "Linux":
        if machine in ("x86_64", "amd64"):
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        elif machine in ("aarch64", "arm64"):
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64"
        else:
            raise RuntimeError(f"Unsupported: {machine}")
    elif system == "Darwin":
        if machine in ("x86_64", "amd64"):
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64"
        elif machine in ("aarch64", "arm64"):
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-arm64"
        else:
            raise RuntimeError(f"Unsupported: {machine}")
    elif system == "Windows":
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    else:
        raise RuntimeError(f"Unsupported OS: {system}")
    
    binary_name = "cloudflared.exe" if system == "Windows" else "cloudflared"
    binary_path = os.path.join(os.getcwd(), binary_name)
    
    if os.path.exists(binary_path):
        print("✅ cloudflared already downloaded.")
        return binary_path
    
    print(f"⬇️ Downloading from: {url}")
    
    try:
        urllib.request.urlretrieve(url, binary_path)
        if system != "Windows":
            st = os.stat(binary_path)
            os.chmod(binary_path, st.st_mode | stat.S_IEXEC)
        print("✅ cloudflared downloaded successfully.")
        return binary_path
    except Exception as e:
        print(f"❌ Failed to download cloudflared: {e}")
        return None


def start_server():
    """Start the FastAPI worker server"""
    print("🚀 Starting worker server...")
    
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    
    from app.worker_server import build_fastapi_app
    import uvicorn
    
    port = int(os.environ.get("WORKER_PORT", 8000))
    
    def run_server():
        app = build_fastapi_app()
        uvicorn.run(app, host="0.0.0.0", port=port)
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    time.sleep(2)  # Wait for server to start
    print(f"✅ Worker server running on http://localhost:{port}")
    return port


def start_tunnel(port=8000):
    """Start cloudflared tunnel and print the URL"""
    binary_name = "cloudflared.exe" if platform.system() == "Windows" else "cloudflared"
    binary_path = os.path.join(os.getcwd(), binary_name)
    
    if not os.path.exists(binary_path):
        print("❌ cloudflared not found. Downloading...")
        binary_path = download_cloudflared()
        if not binary_path:
            print("❌ Failed to download cloudflared.")
            return
    
    print("🔗 Starting cloudflared tunnel...")
    
    cmd = [binary_path, "tunnel", "--url", f"http://localhost:{port}"]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        print("\n" + "="*60)
        print("🔍 Looking for tunnel URL...")
        print("="*60 + "\n")
        
        tunnel_url = None
        for line in iter(process.stdout.readline, ''):
            print(line.strip())
            
            if "https://" in line and ".trycloudflare.com" in line:
                import re
                match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
                if match:
                    tunnel_url = match.group(0)
                    print("\n" + "="*60)
                    print(f"✅ TUNNEL URL: {tunnel_url}")
                    print("="*60)
                    print("\n📋 Copy this URL and paste it into the Main app's Workers tab.")
                    print("="*60 + "\n")
        
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping tunnel...")
        process.terminate()
    except Exception as e:
        print(f"❌ Tunnel error: {e}")


def main():
    print("""
    ╔═══════════════════════════════════════════╗
    ║   🎬 Universal Documentary Studio        ║
    ║   GPU Worker                             ║
    ╚═══════════════════════════════════════════╝
    """)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            print("⚠️ No GPU detected. Running on CPU.")
    except:
        print("⚠️ PyTorch not installed yet.")
    
    if not install_requirements():
        sys.exit(1)
    
    port = start_server()
    
    download_cloudflared()
    start_tunnel(port)


if __name__ == "__main__":
    main()
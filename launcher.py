#!/usr/bin/env python3
"""
Model Launcher (Gradio version)
--------------------------------
A web-based UI for managing local LLM models via Ollama. Works in Google Colab,
Jupyter, or any environment without a display server (Tkinter needs a display;
this doesn't).

Features:
  - Lists available models with their approximate download size
  - Per-model "Install" button that pulls the model and streams live logs
  - Per-model "Delete" button (enabled once a model is installed)
  - Status shows Not Installed / Installing... / Installed

Requirements:
  - pip install gradio
  - Ollama installed and on PATH: https://ollama.com
    In Colab, install it first with:
      !curl -fsSL https://ollama.com/install.sh | sh
      !nohup ollama serve > ollama.log 2>&1 &

Run:
  python3 launcher.py
"""

import subprocess
import shutil
import gradio as gr

# ----------------------------------------------------------------------------
# Model catalog: (display name, ollama tag, approximate download size)
# Edit this list to add/remove models you want available in the launcher.
# ----------------------------------------------------------------------------
MODELS = [
    {"name": "Llama 3.2 1B",     "tag": "llama3.2:1b",     "size": "1.3 GB"},
    {"name": "Llama 3.2 3B",     "tag": "llama3.2:3b",     "size": "2.0 GB"},
    {"name": "Llama 3.1 8B",     "tag": "llama3.1:8b",     "size": "4.7 GB"},
    {"name": "Mistral 7B",       "tag": "mistral:7b",      "size": "4.1 GB"},
    {"name": "Gemma 2 2B",       "tag": "gemma2:2b",       "size": "1.6 GB"},
    {"name": "Gemma 2 9B",       "tag": "gemma2:9b",       "size": "5.4 GB"},
    {"name": "Phi-3 Mini",       "tag": "phi3:mini",       "size": "2.3 GB"},
    {"name": "Qwen 2.5 7B",      "tag": "qwen2.5:7b",      "size": "4.7 GB"},
    {"name": "CodeLlama 7B",     "tag": "codellama:7b",    "size": "3.8 GB"},
    {"name": "DeepSeek R1 7B",   "tag": "deepseek-r1:7b",  "size": "4.7 GB"},
]

OLLAMA_AVAILABLE = shutil.which("ollama") is not None


def ollama_missing_notice():
    if OLLAMA_AVAILABLE:
        return ""
    return (
        "⚠️ **Ollama not found on PATH.** Install it first, e.g. in Colab:\n\n"
        "```\n!curl -fsSL https://ollama.com/install.sh | sh\n"
        "!nohup ollama serve > ollama.log 2>&1 &\n```"
    )


def is_installed(tag):
    if not OLLAMA_AVAILABLE:
        return False
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        base_name = tag.split(":")[0]
        return tag in result.stdout or base_name in result.stdout
    except Exception:
        return False


def status_label(tag):
    return "🟢 Installed" if is_installed(tag) else "⚪ Not Installed"


def install_model(tag, log_text):
    """Generator: streams `ollama pull` output live and updates status/buttons."""
    if not OLLAMA_AVAILABLE:
        new_log = (log_text or "") + f"\n[{tag}] Error: 'ollama' command not found.\n"
        yield new_log, status_label(tag), gr.update(interactive=True), gr.update(interactive=False)
        return

    log_text = (log_text or "") + f"\n$ ollama pull {tag}\n"
    yield log_text, "🟡 Installing...", gr.update(interactive=False), gr.update(interactive=False)

    try:
        process = subprocess.Popen(
            ["ollama", "pull", tag],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            log_text += line
            yield log_text, "🟡 Installing...", gr.update(interactive=False), gr.update(interactive=False)
        process.wait()
        success = process.returncode == 0
    except Exception as e:
        log_text += f"Error: {e}\n"
        success = False

    if success:
        log_text += f"[{tag}] Install complete.\n"
        yield log_text, "🟢 Installed", gr.update(interactive=False), gr.update(interactive=True)
    else:
        log_text += f"[{tag}] Install failed.\n"
        yield log_text, "⚪ Not Installed", gr.update(interactive=True), gr.update(interactive=False)


def delete_model(tag, log_text):
    log_text = (log_text or "") + f"\n$ ollama rm {tag}\n"
    try:
        result = subprocess.run(["ollama", "rm", tag], capture_output=True, text=True, timeout=30)
        log_text += (result.stdout or "") + (result.stderr or "")
        success = result.returncode == 0
    except Exception as e:
        log_text += f"Error: {e}\n"
        success = False

    if success:
        log_text += f"[{tag}] Delete complete.\n"
        return log_text, "⚪ Not Installed", gr.update(interactive=True), gr.update(interactive=False)
    else:
        log_text += f"[{tag}] Delete failed.\n"
        return log_text, "🟢 Installed", gr.update(interactive=False), gr.update(interactive=True)


def clear_logs():
    return ""


with gr.Blocks(title="Model Launcher") as demo:
    gr.Markdown("# 🚀 Model Launcher")
    gr.Markdown(ollama_missing_notice())

    log_box = gr.Textbox(
        label="Installation Logs",
        lines=14,
        max_lines=14,
        interactive=False,
        autoscroll=True,
    )
    clear_btn = gr.Button("Clear logs", size="sm")
    clear_btn.click(fn=clear_logs, outputs=log_box)

    gr.Markdown("---")

    for model in MODELS:
        tag = model["tag"]
        with gr.Row(equal_height=True):
            gr.Markdown(f"**{model['name']}**", elem_id=f"name-{tag}")
            gr.Markdown(model["size"])
            status_box = gr.Markdown(status_label(tag))
            install_btn = gr.Button("Install", size="sm", interactive=not is_installed(tag))
            delete_btn = gr.Button("Delete", size="sm", interactive=is_installed(tag))

        install_btn.click(
            fn=install_model,
            inputs=[gr.State(tag), log_box],
            outputs=[log_box, status_box, install_btn, delete_btn],
        )
        delete_btn.click(
            fn=delete_model,
            inputs=[gr.State(tag), log_box],
            outputs=[log_box, status_box, install_btn, delete_btn],
        )


if __name__ == "__main__":
    # share=True gives you a public link, which is what you need in Colab.
    demo.launch(share=True)

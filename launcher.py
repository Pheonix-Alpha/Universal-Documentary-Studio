#!/usr/bin/env python3
"""
Launcher
--------
A web-based dev console for Colab/Jupyter (no display server needed).

Three tabs:
  1. Models    - list of LLM models with size, Install (streams logs), Delete
  2. Terminal  - run any shell command, streamed output, persistent cwd (cd works)
  3. Files     - browse a directory, delete files/folders

Key detail: this whole app runs as one long-lived Python process. If you install
a new binary (like `ollama`) via the Terminal *after* this process has already
started, that binary won't automatically be visible to Python's `PATH` unless
Python re-checks. So instead of caching "is ollama available" once at startup,
every relevant function re-scans PATH + common install locations on demand, and
using the Terminal automatically refreshes the Models tab's status afterwards.

Requirements:
  - pip install gradio
  - (Models tab) Ollama installed and on PATH: https://ollama.com

Run:
  python3 launcher.py
"""

import os
import shutil
import subprocess
import gradio as gr

# ============================================================================
# Shared helpers
# ============================================================================

def human_size(num_bytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


# Directories commonly used by install scripts, checked in addition to PATH.
_CANDIDATE_BIN_DIRS = [
    "/usr/local/bin",
    "/usr/bin",
    "/opt/ollama/bin",
    "/opt/homebrew/bin",
    os.path.expanduser("~/.ollama/bin"),
    os.path.expanduser("~/bin"),
    os.path.expanduser("~/.local/bin"),
]


def ensure_on_path(executable):
    """
    Return True if `executable` is runnable. Re-checks PATH fresh every call
    (doesn't trust a cached result), and if the binary exists in a well-known
    install directory that isn't currently on this process's PATH, adds that
    directory to os.environ['PATH'] so subprocess calls can find it without
    needing a full restart of this Python process.
    """
    if shutil.which(executable):
        return True
    for d in _CANDIDATE_BIN_DIRS:
        exe = os.path.join(d, executable)
        if os.path.isfile(exe) and os.access(exe, os.X_OK):
            current = os.environ.get("PATH", "")
            if d not in current.split(os.pathsep):
                os.environ["PATH"] = d + os.pathsep + current
            return True
    return False


# ============================================================================
# Tab 1: Models (Ollama)
# ============================================================================

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


def ollama_ready():
    return ensure_on_path("ollama")


def ollama_notice():
    if ollama_ready():
        return "✅ **Ollama detected and ready.**"
    return (
        "⚠️ **Ollama not found yet.** Click **Install Ollama** below, or run the "
        "install command in the Terminal tab — either way this tab updates automatically."
    )


def is_installed(tag):
    if not ollama_ready():
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
    if not ollama_ready():
        new_log = (log_text or "") + f"\n[{tag}] Error: 'ollama' not found. Install it first (see notice above).\n"
        yield new_log, status_label(tag), gr.update(interactive=True), gr.update(interactive=False)
        return

    log_text = (log_text or "") + f"\n$ ollama pull {tag}\n"
    yield log_text, "🟡 Installing...", gr.update(interactive=False), gr.update(interactive=False)

    try:
        process = subprocess.Popen(
            ["ollama", "pull", tag],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
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


def clear_model_logs():
    return ""


def install_ollama_binary(log_text):
    """Runs the official Ollama install script, streamed into the model log."""
    log_text = (log_text or "") + "\n$ curl -fsSL https://ollama.com/install.sh | sh\n"
    yield log_text
    try:
        process = subprocess.Popen(
            "curl -fsSL https://ollama.com/install.sh | sh",
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in process.stdout:
            log_text += line
            yield log_text
        process.wait()
    except Exception as e:
        log_text += f"Error: {e}\n"
        yield log_text
        return

    if ensure_on_path("ollama"):
        log_text += "\nOllama installed and detected. Starting server...\n"
        yield log_text
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            log_text += "Ollama server started in the background.\n"
        except Exception as e:
            log_text += f"Could not start server automatically: {e}\n"
    else:
        log_text += "\nInstall finished but 'ollama' still not found on PATH. Check the log above.\n"
    yield log_text


# ============================================================================
# Tab 2: Terminal (arbitrary shell commands, streamed, persistent cwd)
# ============================================================================

def run_command(cmd, cwd, term_log):
    cmd = (cmd or "").strip()
    term_log = term_log or ""
    if not cmd:
        yield term_log, cwd, cwd
        return

    term_log += f"\n{cwd}$ {cmd}\n"

    if cmd == "cd" or cmd.startswith("cd "):
        target = cmd[2:].strip() or os.path.expanduser("~")
        new_path = os.path.normpath(os.path.join(cwd, os.path.expanduser(target)))
        if os.path.isdir(new_path):
            term_log += f"(changed directory to {new_path})\n"
            yield term_log, new_path, new_path
        else:
            term_log += f"cd: no such directory: {target}\n"
            yield term_log, cwd, cwd
        return

    try:
        process = subprocess.Popen(
            cmd, shell=True, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in process.stdout:
            term_log += line
            yield term_log, cwd, cwd
        process.wait()
        term_log += f"[exit code {process.returncode}]\n"
        yield term_log, cwd, cwd
    except Exception as e:
        term_log += f"Error: {e}\n"
        yield term_log, cwd, cwd


def clear_terminal_log():
    return ""


# ============================================================================
# Tab 3: Files (browse + delete)
# ============================================================================

def list_directory(path):
    path = path or "."
    path = os.path.expanduser(path)
    if not os.path.isdir(path):
        return [["Error", f"Not a directory: {path}", ""]], path

    rows = []
    try:
        entries = sorted(os.listdir(path), key=lambda e: (not os.path.isdir(os.path.join(path, e)), e.lower()))
    except Exception as e:
        return [["Error", str(e), ""]], path

    for entry in entries:
        full = os.path.join(path, entry)
        try:
            if os.path.isdir(full):
                rows.append([entry + "/", "folder", ""])
            else:
                size = os.path.getsize(full)
                rows.append([entry, "file", human_size(size)])
        except Exception:
            rows.append([entry, "?", ""])

    if not rows:
        rows = [["(empty directory)", "", ""]]
    return rows, path


def delete_path(target_path):
    target_path = os.path.expanduser((target_path or "").strip())
    if not target_path:
        return "No path given."
    if not os.path.exists(target_path):
        return f"Path does not exist: {target_path}"
    try:
        if os.path.isdir(target_path):
            shutil.rmtree(target_path)
            return f"Deleted directory: {target_path}"
        else:
            os.remove(target_path)
            return f"Deleted file: {target_path}"
    except Exception as e:
        return f"Error deleting {target_path}: {e}"


# ============================================================================
# UI
# ============================================================================

with gr.Blocks(title="Launcher") as demo:
    gr.Markdown("# 🚀 Launcher")

    # Components that need refreshing whenever ollama's availability might have
    # changed (after Terminal commands, or the dedicated Install button).
    notice_md = None
    model_rows = []  # list of (tag, status_box, install_btn, delete_btn)

    with gr.Tabs():
        # ---------------- Models tab ----------------
        with gr.Tab("Models"):
            notice_md = gr.Markdown(ollama_notice())
            with gr.Row():
                install_ollama_btn = gr.Button("⬇️ Install Ollama", size="sm")

            model_log = gr.Textbox(label="Installation Logs", lines=12, max_lines=12,
                                    interactive=False, autoscroll=True)
            clear_models_btn = gr.Button("Clear logs", size="sm")
            clear_models_btn.click(fn=clear_model_logs, outputs=model_log)

            gr.Markdown("---")

            for model in MODELS:
                tag = model["tag"]
                with gr.Row(equal_height=True):
                    gr.Markdown(f"**{model['name']}**")
                    gr.Markdown(model["size"])
                    status_box = gr.Markdown(status_label(tag))
                    install_btn = gr.Button("Install", size="sm", interactive=not is_installed(tag))
                    delete_btn = gr.Button("Delete", size="sm", interactive=is_installed(tag))

                model_rows.append((tag, status_box, install_btn, delete_btn))

                install_btn.click(
                    fn=install_model,
                    inputs=[gr.State(tag), model_log],
                    outputs=[model_log, status_box, install_btn, delete_btn],
                )
                delete_btn.click(
                    fn=delete_model,
                    inputs=[gr.State(tag), model_log],
                    outputs=[model_log, status_box, install_btn, delete_btn],
                )

        # ---------------- Terminal tab ----------------
        with gr.Tab("Terminal"):
            gr.Markdown(
                "Run any shell command — `pip install ...`, `apt-get install ...`, "
                "`curl -fsSL https://ollama.com/install.sh | sh`, `cd ...`, etc.\n\n"
                "Installing something here (like Ollama) automatically refreshes the Models tab."
            )
            cwd_state = gr.State(os.getcwd())
            cwd_display = gr.Textbox(label="Working directory", value=os.getcwd(), interactive=False)
            term_log = gr.Textbox(label="Terminal", lines=18, max_lines=18,
                                   interactive=False, autoscroll=True)
            with gr.Row():
                cmd_input = gr.Textbox(label="Command", placeholder="pip install numpy", scale=5)
                run_btn = gr.Button("Run", variant="primary", scale=1)
            clear_term_btn = gr.Button("Clear terminal", size="sm")
            clear_term_btn.click(fn=clear_terminal_log, outputs=term_log)

        # ---------------- Files tab ----------------
        with gr.Tab("Files"):
            gr.Markdown("Browse a directory and delete files or folders.")
            with gr.Row():
                dir_input = gr.Textbox(label="Directory", value=os.getcwd(), scale=4)
                list_btn = gr.Button("List", scale=1)

            file_table = gr.Dataframe(
                headers=["Name", "Type", "Size"],
                datatype=["str", "str", "str"],
                interactive=False,
                wrap=True,
            )

            list_btn.click(fn=list_directory, inputs=dir_input, outputs=[file_table, dir_input])
            dir_input.submit(fn=list_directory, inputs=dir_input, outputs=[file_table, dir_input])

            gr.Markdown("---")
            with gr.Row():
                delete_input = gr.Textbox(
                    label="Path to delete (file or folder, full or relative to above)", scale=4
                )
                delete_btn = gr.Button("Delete", variant="stop", scale=1)
            delete_result = gr.Textbox(label="Result", interactive=False)

            delete_btn.click(fn=delete_path, inputs=delete_input, outputs=delete_result)

    # ------------------------------------------------------------------
    # Unified refresh: re-check ollama availability + every model row's
    # install state. Wired to fire after Terminal commands and the
    # dedicated Install Ollama button, so the two tabs stay in sync.
    # ------------------------------------------------------------------
    def refresh_models_tab():
        outputs = [ollama_notice()]
        for tag, *_ in model_rows:
            installed = is_installed(tag)
            outputs.append(status_label(tag))
            outputs.append(gr.update(interactive=not installed))
            outputs.append(gr.update(interactive=installed))
        return outputs

    refresh_outputs = [notice_md]
    for _tag, status_box, install_btn, delete_btn in model_rows:
        refresh_outputs.extend([status_box, install_btn, delete_btn])

    install_ollama_btn.click(
        fn=install_ollama_binary, inputs=model_log, outputs=model_log,
    ).then(fn=refresh_models_tab, outputs=refresh_outputs)

    run_btn.click(
        fn=run_command, inputs=[cmd_input, cwd_state, term_log],
        outputs=[term_log, cwd_state, cwd_display],
    ).then(fn=lambda: "", outputs=cmd_input
    ).then(fn=refresh_models_tab, outputs=refresh_outputs)

    cmd_input.submit(
        fn=run_command, inputs=[cmd_input, cwd_state, term_log],
        outputs=[term_log, cwd_state, cwd_display],
    ).then(fn=lambda: "", outputs=cmd_input
    ).then(fn=refresh_models_tab, outputs=refresh_outputs)


if __name__ == "__main__":
    demo.launch(share=True)

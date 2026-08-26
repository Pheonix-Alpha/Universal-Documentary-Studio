#!/usr/bin/env python3
"""
Model Launcher
--------------
A simple desktop GUI (Tkinter) for managing local LLM models via Ollama.

Features:
  - Lists available models with their approximate download size
  - Per-model "Install" button that pulls the model and streams live logs
  - Per-model "Delete" button (enabled once a model is installed) to remove it
  - Status column shows Not Installed / Installing... / Installed

Requirements:
  - Ollama must be installed and on PATH: https://ollama.com
  - Python 3.8+

Run:
  python3 launcher.py
"""

import subprocess
import threading
import queue
import shutil
import tkinter as tk
from tkinter import ttk, messagebox

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


class ModelLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Model Launcher")
        self.geometry("880x560")
        self.minsize(760, 480)

        self.log_queue = queue.Queue()
        self.row_widgets = {}   # tag -> dict of widgets for that row
        self.installing = set()  # tags currently installing

        self._check_ollama()
        self._build_ui()
        self._refresh_installed_status()
        self.after(100, self._poll_log_queue)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _check_ollama(self):
        if shutil.which("ollama") is None:
            messagebox.showwarning(
                "Ollama not found",
                "The 'ollama' command was not found on your PATH.\n\n"
                "Install it from https://ollama.com to use this launcher.\n"
                "The UI will still open, but installs/deletes will fail.",
            )

    def _build_ui(self):
        # Header
        header = ttk.Frame(self, padding=(12, 10))
        header.pack(fill="x")
        ttk.Label(header, text="Model Launcher", font=("Segoe UI", 16, "bold")).pack(side="left")
        ttk.Button(header, text="Refresh status", command=self._refresh_installed_status).pack(side="right")

        # Table header
        cols = ("Model", "Size", "Status")
        table_header = ttk.Frame(self, padding=(12, 0))
        table_header.pack(fill="x")
        widths = (260, 90, 130)
        for text, w in zip(cols, widths):
            ttk.Label(table_header, text=text, font=("Segoe UI", 10, "bold"), width=w // 8).pack(side="left")

        # Scrollable model list
        list_container = ttk.Frame(self, padding=(12, 4))
        list_container.pack(fill="both", expand=False)

        canvas = tk.Canvas(list_container, height=230, highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=canvas.yview)
        self.rows_frame = ttk.Frame(canvas)

        self.rows_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.rows_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for model in MODELS:
            self._add_model_row(model)

        # Log panel
        log_label = ttk.Frame(self, padding=(12, 8, 12, 0))
        log_label.pack(fill="x")
        ttk.Label(log_label, text="Installation Logs", font=("Segoe UI", 11, "bold")).pack(side="left")
        ttk.Button(log_label, text="Clear logs", command=self._clear_logs).pack(side="right")

        log_frame = ttk.Frame(self, padding=12)
        log_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(log_frame, bg="#111318", fg="#d7e0ea", insertbackground="white",
                                 wrap="word", font=("Consolas", 10), state="disabled")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

    def _add_model_row(self, model):
        tag = model["tag"]
        row = ttk.Frame(self.rows_frame, padding=(0, 4))
        row.pack(fill="x")

        name_lbl = ttk.Label(row, text=model["name"], width=32)
        name_lbl.pack(side="left")

        size_lbl = ttk.Label(row, text=model["size"], width=11)
        size_lbl.pack(side="left")

        status_lbl = ttk.Label(row, text="Checking...", width=16, foreground="#888888")
        status_lbl.pack(side="left")

        install_btn = ttk.Button(row, text="Install", command=lambda: self._start_install(tag))
        install_btn.pack(side="left", padx=(4, 4))

        delete_btn = ttk.Button(row, text="Delete", command=lambda: self._start_delete(tag),
                                 state="disabled")
        delete_btn.pack(side="left")

        self.row_widgets[tag] = {
            "status": status_lbl,
            "install_btn": install_btn,
            "delete_btn": delete_btn,
        }

    # ------------------------------------------------------------------
    # Status checking
    # ------------------------------------------------------------------
    def _refresh_installed_status(self):
        if shutil.which("ollama") is None:
            for tag, widgets in self.row_widgets.items():
                widgets["status"].config(text="Ollama missing", foreground="#c0392b")
            return

        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            installed_output = result.stdout
        except Exception:
            installed_output = ""

        for tag, widgets in self.row_widgets.items():
            if tag in self.installing:
                continue  # don't clobber "Installing..." status
            base_name = tag.split(":")[0]
            is_installed = tag in installed_output or base_name in installed_output
            self._set_row_state(tag, "Installed" if is_installed else "Not Installed")

    def _set_row_state(self, tag, status_text):
        widgets = self.row_widgets[tag]
        widgets["status"].config(text=status_text)
        if status_text == "Installed":
            widgets["status"].config(foreground="#2e7d32")
            widgets["install_btn"].config(state="disabled")
            widgets["delete_btn"].config(state="normal")
        elif status_text == "Installing...":
            widgets["status"].config(foreground="#e08e00")
            widgets["install_btn"].config(state="disabled")
            widgets["delete_btn"].config(state="disabled")
        else:  # Not Installed
            widgets["status"].config(foreground="#888888")
            widgets["install_btn"].config(state="normal")
            widgets["delete_btn"].config(state="disabled")

    # ------------------------------------------------------------------
    # Install
    # ------------------------------------------------------------------
    def _start_install(self, tag):
        if tag in self.installing:
            return
        if shutil.which("ollama") is None:
            messagebox.showerror("Ollama not found", "Install Ollama first: https://ollama.com")
            return

        self.installing.add(tag)
        self._set_row_state(tag, "Installing...")
        self._log(f"\n$ ollama pull {tag}\n")

        thread = threading.Thread(target=self._run_install, args=(tag,), daemon=True)
        thread.start()

    def _run_install(self, tag):
        try:
            process = subprocess.Popen(
                ["ollama", "pull", tag],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in process.stdout:
                self.log_queue.put((tag, line))
            process.wait()
            success = process.returncode == 0
        except Exception as e:
            self.log_queue.put((tag, f"Error: {e}\n"))
            success = False

        self.log_queue.put((tag, "__DONE__:INSTALL:" + ("OK" if success else "FAIL")))

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------
    def _start_delete(self, tag):
        if not messagebox.askyesno("Confirm delete", f"Delete model '{tag}' from disk?"):
            return
        self._log(f"\n$ ollama rm {tag}\n")
        thread = threading.Thread(target=self._run_delete, args=(tag,), daemon=True)
        thread.start()

    def _run_delete(self, tag):
        try:
            result = subprocess.run(["ollama", "rm", tag], capture_output=True, text=True, timeout=30)
            output = (result.stdout or "") + (result.stderr or "")
            success = result.returncode == 0
        except Exception as e:
            output = f"Error: {e}\n"
            success = False

        self.log_queue.put((tag, output))
        self.log_queue.put((tag, "__DONE__:DELETE:" + ("OK" if success else "FAIL")))

    # ------------------------------------------------------------------
    # Logging / queue polling (thread-safe UI updates)
    # ------------------------------------------------------------------
    def _poll_log_queue(self):
        try:
            while True:
                tag, line = self.log_queue.get_nowait()

                if line.startswith("__DONE__:INSTALL:"):
                    ok = line.endswith("OK")
                    self.installing.discard(tag)
                    self._set_row_state(tag, "Installed" if ok else "Not Installed")
                    self._log(f"[{tag}] {'Install complete.' if ok else 'Install failed.'}\n")
                elif line.startswith("__DONE__:DELETE:"):
                    ok = line.endswith("OK")
                    self._set_row_state(tag, "Not Installed" if ok else "Installed")
                    self._log(f"[{tag}] {'Delete complete.' if ok else 'Delete failed.'}\n")
                else:
                    self._log(f"[{tag}] {line}")
        except queue.Empty:
            pass
        self.after(100, self._poll_log_queue)

    def _log(self, text):
        self.log_text.config(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _clear_logs(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")


if __name__ == "__main__":
    app = ModelLauncher()
    app.mainloop()

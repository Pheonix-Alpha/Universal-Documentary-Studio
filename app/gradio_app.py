import gradio as gr

from app import compute, config
from app import model_manager as mm
from app import worker_client
from app.pipeline import run_pipeline


def _clip_model_choices():
    return [m["id"] for m in compute.list_models() if m["id"].startswith("clip")]


def build_app():
    with gr.Blocks(title="Story -> Verified Image Storyboard") as demo:
        gr.Markdown("# 📖 Story → Verified Image Storyboard")
        gr.Markdown(
            "Splits a story into scenes, generates multiple search queries per scene, "
            "pulls candidates from Wikimedia Commons / Unsplash / the web, and re-ranks "
            "them with CLIP. Set `ANTHROPIC_API_KEY` before launching for smarter scene "
            "and query understanding; otherwise a rule-based fallback is used."
        )

        with gr.Tab("Generate Storyboard"):
            with gr.Row():
                with gr.Column(scale=2):
                    story_box = gr.Textbox(
                        lines=8,
                        label="Story",
                        placeholder="Paste your story here, e.g. 'In 1969, Neil Armstrong stepped onto the surface of the Moon during the Apollo 11 mission...'",
                    )
                    model_dd = gr.Dropdown(
                        choices=_clip_model_choices(),
                        value="clip-vit-b-32",
                        label="CLIP model to use (download it first in the Models tab)",
                    )
                    top_k_slider = gr.Slider(1, 10, value=3, step=1, label="Images to keep per scene")
                    start_btn = gr.Button("▶️ Start", variant="primary")
                with gr.Column(scale=1):
                    stage_label = gr.Textbox(label="Current stage", interactive=False)
                    progress_bar = gr.Slider(0, 100, value=0, label="Progress %", interactive=False)
                    log_box = gr.Textbox(label="Log", interactive=False, lines=5)

            gallery = gr.Gallery(label="Storyboard", columns=3, height=520)

            def _run(story, model_id, top_k):
                for update in run_pipeline(story, clip_model_id=model_id, top_k=int(top_k)):
                    yield update["stage"], update["pct"], update["log"], update["gallery"]

            start_btn.click(
                _run,
                inputs=[story_box, model_dd, top_k_slider],
                outputs=[stage_label, progress_bar, log_box, gallery],
            )

        with gr.Tab("Models"):
            gr.Markdown(
                "### Installed / available models\n"
                "Download only what you need, and delete models any time to free up "
                "Colab disk space."
            )
            refresh_state = gr.State(0)
            status_box = gr.Markdown("")

            @gr.render(inputs=refresh_state)
            def render_models(_tick):
                for m in mm.list_models():
                    with gr.Row():
                        with gr.Column(scale=3):
                            status = "✅ Installed" if m["installed"] else "⬜ Not downloaded"
                            gr.Markdown(
                                f"**{m['name']}**  \n{m['description']}  \n~{m['size_mb']} MB · {status}"
                            )
                        with gr.Column(scale=2):
                            if m["installed"]:
                                del_btn = gr.Button("🗑️ Delete", size="sm", variant="stop")

                                def _delete(model_id=m["id"]):
                                    mm.delete_model(model_id)
                                    return f"Deleted **{model_id}**."

                                del_btn.click(_delete, outputs=status_box).then(
                                    lambda v: v + 1, inputs=refresh_state, outputs=refresh_state
                                )
                            else:
                                dl_btn = gr.Button("⬇️ Download", size="sm")
                                dl_progress = gr.Slider(0, 100, value=0, label="Download %", interactive=False)

                                def _download(model_id=m["id"]):
                                    for pct, msg in mm.download_model_stream(model_id):
                                        if pct is None:
                                            yield gr.update(), msg
                                        else:
                                            yield gr.update(value=pct), msg

                                dl_btn.click(_download, outputs=[dl_progress, status_box]).then(
                                    lambda v: v + 1, inputs=refresh_state, outputs=refresh_state
                                )

        with gr.Tab("⚙️ Workers (GPU offload)"):
            gr.Markdown(
                "### Optional remote GPU workers\n"
                "Run `python worker.py` in one or more **separate** Colab notebooks "
                "(each with its own GPU runtime), copy the URL each one prints, and "
                "add it below. You can connect several at once -- storyboard generation "
                "automatically round-robins CLIP ranking across every connected worker "
                "instead of running locally, and falls back to the next worker if one "
                "drops."
            )
            with gr.Row():
                worker_url_box = gr.Textbox(
                    label="Worker URL",
                    placeholder="https://xxxx-xxxx.trycloudflare.com",
                    scale=3,
                )
                worker_label_box = gr.Textbox(
                    label="Label (optional)",
                    placeholder="e.g. 'Colab A'",
                    scale=2,
                )
                add_worker_btn = gr.Button("➕ Add worker", scale=1, variant="primary")

            workers_refresh = gr.State(0)
            add_worker_status = gr.Markdown("")
            worker_model_status = gr.Markdown("")

            def _add_worker(url, label):
                try:
                    worker_client.add_worker(url, label)
                    return "", "", ""
                except Exception as e:  # noqa: BLE001
                    return gr.update(), gr.update(), f"⚠️ {e}"

            add_worker_btn.click(
                _add_worker,
                inputs=[worker_url_box, worker_label_box],
                outputs=[worker_url_box, worker_label_box, add_worker_status],
            ).then(lambda v: v + 1, inputs=workers_refresh, outputs=workers_refresh)

            @gr.render(inputs=workers_refresh)
            def render_workers(_tick):
                workers = worker_client.list_workers()
                if not workers:
                    gr.Markdown("_No workers added yet. Paste a worker URL above and click Add worker._")
                    return
                for w in workers:
                    icon = "🟢" if w["connected"] else "🔴"
                    with gr.Accordion(f"{icon} {w['label']} — {w['status']}", open=False):
                        gr.Markdown(f"`{w['url']}`")
                        remove_btn = gr.Button("🗑️ Remove worker", size="sm", variant="stop")

                        def _remove(worker_id=w["id"]):
                            worker_client.remove_worker(worker_id)

                        remove_btn.click(_remove, outputs=[]).then(
                            lambda v: v + 1, inputs=workers_refresh, outputs=workers_refresh
                        )

                        if not w["connected"]:
                            gr.Markdown("_Not reachable right now -- model list unavailable._")
                            continue

                        models = worker_client.list_models(w["id"])
                        if not models:
                            gr.Markdown("_Connected, but couldn't fetch its model list._")
                            continue
                        for m in models:
                            with gr.Row():
                                with gr.Column(scale=3):
                                    status = "✅ Installed" if m["installed"] else "⬜ Not downloaded"
                                    gr.Markdown(
                                        f"**{m['name']}**  \n{m['description']}  \n~{m['size_mb']} MB · {status}"
                                    )
                                with gr.Column(scale=2):
                                    if m["installed"]:
                                        wdel_btn = gr.Button("🗑️ Delete", size="sm", variant="stop")

                                        def _wdelete(worker_id=w["id"], model_id=m["id"], label=w["label"]):
                                            worker_client.delete_model(worker_id, model_id)
                                            return f"Deleted **{model_id}** on {label}."

                                        wdel_btn.click(_wdelete, outputs=worker_model_status).then(
                                            lambda v: v + 1, inputs=workers_refresh, outputs=workers_refresh
                                        )
                                    else:
                                        wdl_btn = gr.Button("⬇️ Download", size="sm")
                                        wdl_progress = gr.Slider(0, 100, value=0, label="Download %", interactive=False)

                                        def _wdownload(worker_id=w["id"], model_id=m["id"]):
                                            for pct, msg in worker_client.download_model_stream(worker_id, model_id):
                                                if pct is None:
                                                    yield gr.update(), msg
                                                else:
                                                    yield gr.update(value=pct), msg

                                        wdl_btn.click(_wdownload, outputs=[wdl_progress, worker_model_status]).then(
                                            lambda v: v + 1, inputs=workers_refresh, outputs=workers_refresh
                                        )

        with gr.Tab("🔑 API Keys"):
            gr.Markdown(
                "### Optional API keys\n"
                "The app works without any of these (rule-based scene splitting, and "
                "the no-key image sources only). Paste a key and hit Save -- it applies "
                "immediately, no restart needed. Keys are kept in memory for this "
                "session only, not written to disk."
            )
            keys_refresh = gr.State(0)

            @gr.render(inputs=keys_refresh)
            def render_keys(_tick):
                for spec in config.KEY_SPECS:
                    key_id = spec["id"]
                    is_set = config.is_key_set(key_id)
                    with gr.Row():
                        with gr.Column(scale=2):
                            status = "🟢 Connected" if is_set else "⬜ Not set"
                            gr.Markdown(
                                f"**{spec['label']}** · {status}  \n{spec['note']}  \n"
                                f"[Get a key]({spec['signup_url']})"
                            )
                        with gr.Column(scale=3):
                            key_box = gr.Textbox(
                                show_label=False,
                                type="password",
                                placeholder="Key is set (paste a new one to replace it)" if is_set else "Paste API key here",
                            )
                            with gr.Row():
                                save_btn = gr.Button("💾 Save", size="sm", variant="primary")
                                clear_btn = gr.Button("Clear", size="sm") if is_set else None

                            def _save(value, key_id=key_id):
                                config.set_key(key_id, value)
                                return ""

                            save_btn.click(_save, inputs=key_box, outputs=key_box).then(
                                lambda v: v + 1, inputs=keys_refresh, outputs=keys_refresh
                            )

                            if clear_btn is not None:
                                def _clear(key_id=key_id):
                                    config.set_key(key_id, "")
                                    return ""

                                clear_btn.click(_clear, outputs=key_box).then(
                                    lambda v: v + 1, inputs=keys_refresh, outputs=keys_refresh
                                )

    return demo


if __name__ == "__main__":
    build_app().queue().launch()
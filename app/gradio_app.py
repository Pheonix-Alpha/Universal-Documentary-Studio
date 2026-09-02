import gradio as gr

from app import compute
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

        with gr.Tab("⚙️ Worker (GPU offload)"):
            gr.Markdown(
                "### Optional remote GPU worker\n"
                "Run `python worker.py` in a **separate** Colab notebook (give that one "
                "a GPU runtime), copy the URL it prints, and paste it below. Once "
                "connected, storyboard generation automatically runs CLIP ranking on "
                "that worker's GPU instead of this notebook -- manage its models here."
            )
            with gr.Row():
                worker_url_box = gr.Textbox(
                    label="Worker URL",
                    placeholder="https://xxxx-xxxx.trycloudflare.com",
                    value=worker_client.get_worker_url() or "",
                    scale=4,
                )
                connect_btn = gr.Button("🔌 Connect", scale=1, variant="primary")
                disconnect_btn = gr.Button("Disconnect", scale=1)

            initial_connected, initial_msg = worker_client.is_connected()
            worker_status = gr.Markdown(f"{'🟢' if initial_connected else '🔴'} {initial_msg}")
            worker_refresh = gr.State(0)
            worker_model_status = gr.Markdown("")

            def _connect(url):
                worker_client.set_worker_url(url)
                ok, msg = worker_client.is_connected()
                return f"{'🟢' if ok else '🔴'} {msg}"

            def _disconnect():
                worker_client.set_worker_url(None)
                return "🔴 Disconnected."

            connect_btn.click(_connect, inputs=worker_url_box, outputs=worker_status).then(
                lambda v: v + 1, inputs=worker_refresh, outputs=worker_refresh
            )
            disconnect_btn.click(_disconnect, outputs=worker_status).then(
                lambda v: v + 1, inputs=worker_refresh, outputs=worker_refresh
            )

            gr.Markdown("#### Worker models")

            @gr.render(inputs=worker_refresh)
            def render_worker_models(_tick):
                connected, _msg = worker_client.is_connected()
                if not connected:
                    gr.Markdown("_Connect to a worker above to see and manage its models._")
                    return
                models = worker_client.list_models()
                if not models:
                    gr.Markdown("_Connected, but couldn't fetch the worker's model list._")
                    return
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

                                def _wdelete(model_id=m["id"]):
                                    worker_client.delete_model(model_id)
                                    return f"Deleted **{model_id}** on the worker."

                                wdel_btn.click(_wdelete, outputs=worker_model_status).then(
                                    lambda v: v + 1, inputs=worker_refresh, outputs=worker_refresh
                                )
                            else:
                                wdl_btn = gr.Button("⬇️ Download", size="sm")
                                wdl_progress = gr.Slider(0, 100, value=0, label="Download %", interactive=False)

                                def _wdownload(model_id=m["id"]):
                                    for pct, msg in worker_client.download_model_stream(model_id):
                                        if pct is None:
                                            yield gr.update(), msg
                                        else:
                                            yield gr.update(value=pct), msg

                                wdl_btn.click(_wdownload, outputs=[wdl_progress, worker_model_status]).then(
                                    lambda v: v + 1, inputs=worker_refresh, outputs=worker_refresh
                                )

    return demo


if __name__ == "__main__":
    build_app().queue().launch()
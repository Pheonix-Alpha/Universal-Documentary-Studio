import gradio as gr

from app import model_manager as mm
from app.pipeline import run_pipeline


def _clip_model_choices():
    return [m["id"] for m in mm.list_models() if m["id"].startswith("clip")]


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

    return demo


if __name__ == "__main__":
    build_app().queue().launch()

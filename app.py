import gradio as gr
from modules.ui.image_ui import define_generate_ui
from modules.ui.image_with_pose_ui import define_generate_with_pose_ui
from modules.ui.inpaint_ui import define_inpaint_ui

# autopep8: off
from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
# autopep8: on


with gr.Blocks() as demo:
    gr.Markdown("# Experiment with Stable Diffusion V1.5")

    with gr.Tabs():
        with gr.Tab("Generate"):
            define_generate_ui()

        with gr.Tab("Generate with Pose"):
            define_generate_with_pose_ui()

        with gr.Tab("Inpaint"):
            define_inpaint_ui()


if __name__ == "__main__":
    demo.launch()

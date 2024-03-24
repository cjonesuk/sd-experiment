import gradio as gr

from modules.workflows.image import run_generate_image_workflow, run_generate_upscaled_image_workflow

# autopep8: off
from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
# autopep8: on


def define_generate_ui():
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox("", label="Enter a prompt",
                                placeholder="a photo of an astronaut riding a horse on mars", lines=3)
            negative_prompt = gr.Textbox(
                "", label="Enter a negative prompt", placeholder="low quality, lowres", lines=3)

            with gr.Row():
                num_inference_steps = gr.Number(
                    30, label="Steps", minimum=1, maximum=100, step=1)
                guidance_scale = gr.Number(
                    8.0, label="Guidance Scale", minimum=0.0, maximum=20.0, step=0.1)

            generate_image = gr.Button("Generate Image")
            generate_upscaled_image = gr.Button(
                "Generate Upscaled Image")

        with gr.Column():
            with gr.Group():
                result_image = gr.Image(label='Generated Image', )

    generate_image.click(
        run_generate_image_workflow,
        inputs=[prompt, negative_prompt, num_inference_steps, guidance_scale],
        outputs=[result_image])

    generate_upscaled_image.click(
        run_generate_upscaled_image_workflow,
        inputs=[prompt, negative_prompt, num_inference_steps, guidance_scale],
        outputs=[result_image])

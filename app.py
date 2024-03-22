from modules.image_generation_tasks import run_generate_image_workflow, run_generate_upscaled_image_workflow
from modules.image_with_pose_generation_tasks import run_generate_image_with_pose_workflow
from comfy_script.runtime.nodes import *
import gradio as gr

import os

# autopep8: off
from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
# autopep8: on


print(os.getcwd())


def define_generate_with_pose_ui():
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

            with gr.Row():
                pose_image = gr.Image(label='Pose Image', type='filepath')
                face_image = gr.Image(label='Face Image', type='filepath')

            generate_image = gr.Button("Generate Image")
            # generate_upscaled_image = gr.Button(
            #     "Generate Upscaled Image")

        with gr.Column():
            with gr.Group():
                result_image = gr.Image(label='Generated Image', )

    generate_image.click(
        run_generate_image_with_pose_workflow,
        inputs=[prompt, negative_prompt,
                num_inference_steps, guidance_scale, pose_image, face_image],
        outputs=[result_image])

    # generate_upscaled_image.click(
    #     run_generate_upscaled_image_workflow,
    #     inputs=[prompt, negative_prompt, num_inference_steps, guidance_scale],
    #     outputs=[result_image])


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


with gr.Blocks() as demo:
    gr.Markdown("# Experiment with Stable Diffusion V1.5")

    with gr.Tabs():
        with gr.Tab("Generate"):
            define_generate_ui()

        with gr.Tab("Generate with Pose"):
            define_generate_with_pose_ui()


if __name__ == "__main__":
    demo.launch()

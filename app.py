from modules.image_workflow import ImageGenerationStageBuilder, ModelApplyStageBuilder, UpscaleImageStageBuilder, ModelApplyStageInput, UserInput
from comfy_script.runtime.nodes import *
import gradio as gr

import os

# autopep8: off
from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
# autopep8: on


print(os.getcwd())

repo_id = "runwayml/stable-diffusion-v1-5"
cyberrealistic = "sd15\\cyberrealistic_v42.safetensors"
analogMadnessInpainting = "sd15/analogMadness_v70-inpainting.safetensors"
analogMadness = "sd15\\analogMadness_v70.safetensors"
model_path = "models/checkpoints"
model_full_path = model_path + '/' + cyberrealistic


model_apply = ModelApplyStageBuilder()
image_generation = ImageGenerationStageBuilder()
upscaled_image_generation = UpscaleImageStageBuilder()


async def run_generate_image_workflow(prompt, negative_prompt, num_inference_steps, guidance_scale):
    image_batch = None

    user_input = UserInput(prompt, negative_prompt)
    model_input = ModelApplyStageInput(
        analogMadness, num_inference_steps, guidance_scale)

    with Workflow(wait=True, cancel_all=True):
        models = model_apply.apply_workflow(
            user_input,
            model_input)

        output = image_generation.generate_with_latent_upscale(
            model_input,
            models)

        image_batch = SaveImage(output.image, 'PY_ComfyUI')

    result_image = await image_batch.wait().get(0)

    return result_image


async def run_generate_upscaled_image_workflow(prompt, negative_prompt, num_inference_steps, guidance_scale):
    image_batch = None

    user_input = UserInput(prompt, negative_prompt)
    model_input = ModelApplyStageInput(
        analogMadness, num_inference_steps, guidance_scale)

    with Workflow(wait=True, cancel_all=True):
        models = model_apply.apply_workflow(
            user_input,
            model_input)

        output = image_generation.generate_with_latent_upscale(
            model_input,
            models)

        upscale = upscaled_image_generation.upscale_extended(output)

        image_batch = SaveImage(upscale.image, 'PY_ComfyUI')

    result_image = await image_batch.wait().get(0)

    return result_image


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


if __name__ == "__main__":
    demo.launch()

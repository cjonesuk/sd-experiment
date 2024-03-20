from comfy_script.runtime.nodes import *
import gradio as gr

import os

from comfy_script.runtime import *
load()

# autopep8: off
from comfy_script.runtime.nodes import *
# autopep8: on

print(os.getcwd())

repo_id = "runwayml/stable-diffusion-v1-5"
cyberrealistic = "sd15\\cyberrealistic_v42.safetensors"
analogMadnessInpainting = "sd15/analogMadness_v70-inpainting.safetensors"
analogMadness = "sd15\\analogMadness_v70.safetensors"
model_path = "models/checkpoints"
model_full_path = model_path + '/' + cyberrealistic


# image_2 = VAEDecode(upscaled_latent, vae)
# image_batch_2 = SaveImage(image_2, 'PY_ComfyUI')


async def generate_image(prompt, negative_prompt, num_inference_steps, guidance_scale):
    print("Generating image...")

    image_batch = None

    with Workflow(wait=True, cancel_all=True):
        image_batch = generate_image_stage(
            prompt, negative_prompt, num_inference_steps, guidance_scale)

    return await image_batch.wait().get(0)

seed = 156680208700281


def generate_image_stage(prompt, negative_prompt, num_inference_steps: int, guidance_scale):
    # num_inference_steps = 70
    # mid_step = int(num_inference_steps * (2.0 // 3.0))
    model, clip, vae = CheckpointLoaderSimple(analogMadness)
    conditioning = CLIPTextEncode(prompt, clip)
    negative_conditioning = CLIPTextEncode(negative_prompt, clip)
    empty_latent = EmptyLatentImage(512, 512, 1)
    # latent = KSampler(model, 156680208700289, num_inference_steps, guidance_scale,
    #                   'euler', 'normal', conditioning, negative_conditioning, empty_latent, 1)

    latent = KSamplerAdvanced(model=model,
                              noise_seed=seed,
                              steps=70,
                              cfg=guidance_scale,
                              sampler_name='euler',
                              scheduler='normal',
                              positive=conditioning,
                              negative=negative_conditioning,
                              latent_image=empty_latent,
                              end_at_step=40)

    upscaled_latent = LatentUpscaleBy(
        samples=latent,
        upscale_method='nearest-exact',
        scale_by=1.5)

    upscaled_latent = KSamplerAdvanced(model=model,
                                       noise_seed=seed,
                                       steps=70,
                                       cfg=guidance_scale,
                                       sampler_name='euler',
                                       scheduler='normal',
                                       positive=conditioning,
                                       negative=negative_conditioning,
                                       latent_image=upscaled_latent,
                                       start_at_step=41)

    image_1 = VAEDecode(latent, vae)
    image_batch_1 = SaveImage(image_1, 'PY_ComfyUI')

    image_2 = VAEDecode(upscaled_latent, vae)
    image_batch_2 = SaveImage(image_2, 'PY_ComfyUI')

    return image_batch_2


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

            generate = gr.Button("Generate Image")

        res = gr.Image(label="output", )

    generate.click(generate_image, inputs=[
                   prompt, negative_prompt, num_inference_steps, guidance_scale], outputs=res)


with gr.Blocks() as demo:
    gr.Markdown("# Experiment with Stable Diffusion V1.5")

    with gr.Tabs():
        with gr.Tab("Generate"):
            define_generate_ui()


if __name__ == "__main__":
    demo.launch()

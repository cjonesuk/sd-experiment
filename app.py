import gradio as gr
from diffusers import StableDiffusionPipeline
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

import torch
import os

print(torch.cuda.is_available())
# print current directory

print(os.getcwd())

repo_id = "runwayml/stable-diffusion-v1-5" 
cyberrealistic = "sd15/cyberrealistic_v42.safetensors"
model_path = "models/checkpoints"
model_full_path = model_path + '/' + cyberrealistic
 
generator = torch.Generator(device="cuda").manual_seed(8)

def generate_image(prompt, negative_prompt, num_inference_steps, guidance_scale):
    print("Generating image...")

    # euler_scheduler = EulerDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
    # ddpm_scheduler = DDPMScheduler.from_pretrained(repo_id, subfolder="scheduler")

    # pipeline = StableDiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
    pipeline = StableDiffusionPipeline.from_single_file(model_full_path, torch_dtype=torch.float16)
 
    pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

    #pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    #pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
 
    pipeline = pipeline.to("cuda")

    image = pipeline(prompt, negative_prompt=negative_prompt, generator=generator, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]  
        
    # image.save("output/astronaut_rides_horse.png")

    return image

with gr.Blocks() as demo:
    gr.Markdown("# Experiment with Stable Diffusion V1.5")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox("", label="Enter a prompt", placeholder="a photo of an astronaut riding a horse on mars", lines=3)
            negative_prompt = gr.Textbox("", label="Enter a negative prompt", placeholder="low quality, lowres", lines=3)

            with gr.Row():
                num_inference_steps = gr.Number(30, label="Steps", minimum=1, maximum=100, step=1)
                guidance_scale = gr.Number(8.0, label="Guidance Scale", minimum=0.0, maximum=20.0, step=0.1)

            generate = gr.Button("Generate Image")

        res = gr.Image(label="output")

    generate.click(generate_image, inputs=[prompt, negative_prompt, num_inference_steps, guidance_scale], outputs=res)

if __name__ == "__main__":
    demo.launch()
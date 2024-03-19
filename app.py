import gradio as gr
 
import os

from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *

 
print(os.getcwd())

repo_id = "runwayml/stable-diffusion-v1-5" 
cyberrealistic = "sd15\\cyberrealistic_v42.safetensors"
analogMadnessInpainting = "sd15/analogMadness_v70-inpainting.safetensors"
analogMadness = "sd15\\analogMadness_v70.safetensors"
model_path = "models/checkpoints"
model_full_path = model_path + '/' + cyberrealistic
 
async def generate_image(prompt, negative_prompt, num_inference_steps, guidance_scale):
    print("Generating image...")
 
    image_batch = None 

    with Workflow(wait=True, cancel_all=True ):
        model, clip, vae = CheckpointLoaderSimple(analogMadness)
        conditioning = CLIPTextEncode(prompt, clip)
        negative_conditioning = CLIPTextEncode(negative_prompt, clip)
        latent = EmptyLatentImage(512, 512, 1)
        latent = KSampler(model, 156680208700286, num_inference_steps, guidance_scale, 'euler', 'normal', conditioning, negative_conditioning, latent, 1)
        image = VAEDecode(latent, vae)
        image_batch = SaveImage(image, 'PY_ComfyUI') 
  
    return await image_batch.wait().get(0)
  

def define_generate_ui():
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox("", label="Enter a prompt", placeholder="a photo of an astronaut riding a horse on mars", lines=3)
            negative_prompt = gr.Textbox("", label="Enter a negative prompt", placeholder="low quality, lowres", lines=3)

            with gr.Row():
                num_inference_steps = gr.Number(30, label="Steps", minimum=1, maximum=100, step=1)
                guidance_scale = gr.Number(8.0, label="Guidance Scale", minimum=0.0, maximum=20.0, step=0.1)

            generate = gr.Button("Generate Image")

        res = gr.Image(label="output", )

    generate.click(generate_image, inputs=[prompt, negative_prompt, num_inference_steps, guidance_scale], outputs=res)

 

with gr.Blocks() as demo:
    gr.Markdown("# Experiment with Stable Diffusion V1.5")

    with gr.Tabs():
        with gr.Tab("Generate"):
            define_generate_ui()
            
           
if __name__ == "__main__":
    demo.launch()
import gradio as gr

# autopep8: off
from comfy_script.runtime import load
load()
from comfy_script.runtime.nodes import Checkpoints
# autopep8: on


def define_prompt_input_ui():
    with gr.Column():
        prompt = gr.Textbox("", label="Enter a prompt",
                            placeholder="a photo of an astronaut riding a horse on mars", lines=3)
        negative_prompt = gr.Textbox(
            "", label="Enter a negative prompt", placeholder="low quality, lowres", lines=3)

    return prompt, negative_prompt


def define_pose_and_face_input_ui():
    with gr.Row():
        pose_image = gr.Image(label='Pose Image', type='filepath')
        face_image = gr.Image(label='Face Image', type='filepath')

    return pose_image, face_image


def define_model_input_ui():
    with gr.Row():
        checkpoint = gr.Dropdown(
            choices=Checkpoints, value=Checkpoints.sd15_analogMadness_v70, label="Checkpoint")
        num_inference_steps = gr.Number(
            30, label="Steps", minimum=1, maximum=100, step=1)
        guidance_scale = gr.Number(
            8.0, label="Guidance Scale", minimum=0.0, maximum=20.0, step=0.1)

    return checkpoint, num_inference_steps, guidance_scale

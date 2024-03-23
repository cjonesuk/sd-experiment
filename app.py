from modules.image_generation_tasks import run_generate_image_workflow, run_generate_upscaled_image_workflow
from modules.image_with_pose_generation_tasks import GenerateImageWithPoseWorkflowStages, run_generate_image_with_pose_workflow
from comfy_script.runtime.nodes import *
import gradio as gr

import os

from modules.image_workflow_builders import FaceInput, ImageGenerationInput, PoseInput, UserInput


print(os.getcwd())


def subtask_image_generation_input_ui():
    with gr.Row():
        checkpoint = gr.Dropdown(
            choices=Checkpoints, value=Checkpoints.sd15_analogMadness_v70, label="Checkpoint")
        num_inference_steps = gr.Number(
            30, label="Steps", minimum=1, maximum=100, step=1)
        guidance_scale = gr.Number(
            8.0, label="Guidance Scale", minimum=0.0, maximum=20.0, step=0.1)

    return checkpoint, num_inference_steps, guidance_scale


def subtask_prompt_input_ui():
    with gr.Column():
        prompt = gr.Textbox("", label="Enter a prompt",
                            placeholder="a photo of an astronaut riding a horse on mars", lines=3)
        negative_prompt = gr.Textbox(
            "", label="Enter a negative prompt", placeholder="low quality, lowres", lines=3)

    return prompt, negative_prompt


def subtask_pose_face_input_ui():
    with gr.Row():
        pose_image = gr.Image(label='Pose Image', type='filepath')
        face_image = gr.Image(label='Face Image', type='filepath')

    return pose_image, face_image


async def handle_pose_workflow_click(
        stage: GenerateImageWithPoseWorkflowStages,
        prompt,
        negative_prompt,
        checkpoint,
        num_inference_steps,
        guidance_scale,
        pose_image: Image,
        face_image: Image):
    user_input = UserInput(prompt, negative_prompt)

    model_input = ImageGenerationInput(
        checkpoint,
        num_inference_steps,
        guidance_scale)

    pose_input = PoseInput(pose_image)
    face_input = FaceInput(face_image)

    return await run_generate_image_with_pose_workflow(
        stage,
        user_input,
        model_input,
        pose_input,
        face_input)


def define_generate_with_pose_ui():
    with gr.Row():
        stage1_state = gr.State(
            value=GenerateImageWithPoseWorkflowStages.GENERATE_IMAGE)
        stage2_state = gr.State(
            value=GenerateImageWithPoseWorkflowStages.UPSCALE_IMAGE)

        with gr.Column():
            with gr.Accordion(label='Image Properties', open=False):
                checkpoint, num_inference_steps, guidance_scale = subtask_image_generation_input_ui()

            with gr.Accordion(label='Input'):
                with gr.Column():
                    pose_image, face_image = subtask_pose_face_input_ui()
                    prompt, negative_prompt = subtask_prompt_input_ui()

            generate_image = gr.Button("Generate Image")
            generate_upscaled_image = gr.Button(
                "Generate Upscaled Image")

        with gr.Column():
            with gr.Group():
                result_image = gr.Image(label='Generated Image', )

    generate_image.click(
        handle_pose_workflow_click,
        inputs=[stage1_state,
                prompt, negative_prompt,
                checkpoint, num_inference_steps, guidance_scale,
                pose_image, face_image],
        outputs=[result_image])

    generate_upscaled_image.click(
        handle_pose_workflow_click,
        inputs=[stage2_state,
                prompt, negative_prompt,
                checkpoint, num_inference_steps, guidance_scale,
                pose_image, face_image],
        outputs=[result_image])


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

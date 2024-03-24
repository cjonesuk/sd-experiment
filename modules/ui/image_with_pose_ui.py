import gradio as gr
from modules.ui.inputs import define_pose_and_face_input_ui, define_prompt_input_ui, define_model_input_ui
from modules.types import FaceInput, ImageGenerationInput, PoseInput, UserInput
from modules.workflows.image_with_pose import GenerateImageWithPoseWorkflowStages, run_generate_image_with_pose_workflow

# autopep8: off
from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
# autopep8: on


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
                checkpoint, num_inference_steps, guidance_scale = define_model_input_ui()

            with gr.Accordion(label='Input'):
                with gr.Column():
                    pose_image, face_image = define_pose_and_face_input_ui()
                    prompt, negative_prompt = define_prompt_input_ui()

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

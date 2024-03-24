import gradio as gr
from modules.ui.inputs import define_prompt_input_ui, define_model_input_ui
from modules.types import ImageGenerationInput, InpaintInput, UserInput
from modules.workflows.inpainting import InpaintStages, run_inpaint_workflow

# autopep8: off
from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
# autopep8: on


async def handle_begin_click(input_image):
    return {
        "background": input_image,
        'layers': [],
        'composite': None
    }


async def handle_inpaint_click(
        stage: InpaintStages,
        prompt,
        negative_prompt,
        checkpoint,
        num_inference_steps,
        guidance_scale,
        input_image_mask):
    user_input = UserInput(prompt, negative_prompt)

    model_input = ImageGenerationInput(
        checkpoint,
        num_inference_steps,
        guidance_scale)

    image = input_image_mask["background"]
    image_mask = input_image_mask['layers'][0]

    inpaint_input = InpaintInput(image, image_mask)

    return await run_inpaint_workflow(
        stage,
        user_input,
        model_input,
        inpaint_input)


def define_inpaint_ui():
    with gr.Row():
        inpaint_stage = gr.State(value=InpaintStages.INPAINT)

        with gr.Column():
            with gr.Accordion(label='Input Image', open=True):
                with gr.Column():
                    input_image = gr.Image(
                        label='Input Image', type='filepath')
                    begin_input_image = gr.Button(value="Begin Input Image")

            with gr.Accordion(label='Image Properties', open=False):
                checkpoint, num_inference_steps, guidance_scale = define_model_input_ui()

            with gr.Accordion(label='Input', open=False):
                with gr.Column():
                    input_image_mask = gr.ImageMask(
                        label='Input Image', type='filepath')
                    prompt, negative_prompt = define_prompt_input_ui()

            inpaint_image = gr.Button("Inpaint")

        with gr.Column():
            with gr.Group():
                result_image = gr.Image(label='Generated Image', )

    begin_input_image.click(
        handle_begin_click,
        inputs=[input_image],
        outputs=[input_image_mask]
    )

    inpaint_image.click(
        handle_inpaint_click,
        inputs=[inpaint_stage,
                prompt, negative_prompt,
                checkpoint, num_inference_steps, guidance_scale,
                input_image_mask],
        outputs=[result_image])
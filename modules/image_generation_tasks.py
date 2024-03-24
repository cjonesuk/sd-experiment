from modules.ImageGenerationStageBuilder import ImageGenerationStageBuilder
from modules.ModelApplyStageBuilder import ModelApplyStageBuilder
from modules.UpscaleImageStageBuilder import UpscaleImageStageBuilder
from modules.types import ImageGenerationInput, UserInput

# autopep8: off
from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
# autopep8: on

analogMadness = "sd15\\analogMadness_v70.safetensors"

model_apply = ModelApplyStageBuilder()
image_generation = ImageGenerationStageBuilder()
upscaled_image_generation = UpscaleImageStageBuilder()


async def run_generate_image_workflow(prompt, negative_prompt, num_inference_steps, guidance_scale):
    image_batch = None

    user_input = UserInput(prompt, negative_prompt)
    model_input = ImageGenerationInput(
        analogMadness, num_inference_steps, guidance_scale)

    with Workflow(wait=True, cancel_all=True):
        models = model_apply.load(
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
    model_input = ImageGenerationInput(
        analogMadness, num_inference_steps, guidance_scale)

    with Workflow(wait=True, cancel_all=True):
        models = model_apply.load(
            user_input,
            model_input)

        output = image_generation.generate_with_latent_upscale(
            model_input,
            models)

        upscale = upscaled_image_generation.upscale_extended(output)

        image_batch = SaveImage(upscale.image, 'PY_ComfyUI')

    result_image = await image_batch.wait().get(0)

    return result_image

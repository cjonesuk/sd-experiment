
from strenum import StrEnum
from modules.image_workflow_builders import (FaceIdApplyStageBuilder,
                                             FaceInput,
                                             FacePreparationStageBuilder,
                                             ImageGenerationStageBuilder,
                                             ImageSaveStageBuilder,
                                             ImageStageOutput,
                                             InpaintInput,
                                             ModelApplyStageBuilder,
                                             PoseApplyStageBuilder,
                                             PoseInput,
                                             UpscaleImageStageBuilder,
                                             ImageGenerationInput,
                                             UserInput
                                             )
# autopep8: off
from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
# autopep8: on


model_apply = ModelApplyStageBuilder()
poses = PoseApplyStageBuilder()
image_generation = ImageGenerationStageBuilder()
upscaled_image_generation = UpscaleImageStageBuilder()
image_saving = ImageSaveStageBuilder()


class InpaintStages(StrEnum):
    INPAINT = 'inpaint'


async def run_inpaint_workflow(
        stage: InpaintStages,
        user_input: UserInput,
        model_input: ImageGenerationInput,
        inpaint_input: InpaintInput,
):
    print('run_inpaint_workflow at stage:', stage)

    with Workflow(wait=True, cancel_all=True) as wf:
        models = model_apply.load(
            user_input,
            model_input)

        input_image, _ = LoadImageFromPath(image=inpaint_input.image_path)
        input_mask_image = LoadImageMask(image=inpaint_input.mask_path)

        input_mask_image_inverted = InvertMask(mask=input_mask_image)

        resized_image, resized_mask = ImageResize(pixels=input_image,
                                                  mask_optional=input_mask_image_inverted,
                                                  action='resize only',
                                                  smaller_side=0,
                                                  larger_side=768,
                                                  scale_factor=0.0,
                                                  resize_mode='any',
                                                  crop_pad_position=0.5,
                                                  pad_feathering=20)

        latent = VAEEncodeForInpaint(pixels=resized_image,
                                     mask=resized_mask,
                                     vae=models.vae,
                                     grow_mask_by=20)

        image_output = image_generation.inpaint(model_input=model_input,
                                                models=models,
                                                latent_input=latent)

        image_batch = SaveImage(
            images=image_output.image, filename_prefix='test')

    print(wf.api_format_json())
    result_image = await image_batch.wait().get(0)

    return result_image

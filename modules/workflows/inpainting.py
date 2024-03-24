
from strenum import StrEnum
from custom_nodes.comfyui_controlnet_aux.src.controlnet_aux.mesh_graphormer.depth_preprocessor import Preprocessor
from modules.workflow_builders.FaceIdApplyStageBuilder import FaceIdApplyStageBuilder
from modules.workflow_builders.FacePreparationStageBuilder import FacePreparationStageBuilder
from modules.workflow_builders.ImageGenerationStageBuilder import ImageGenerationStageBuilder
from modules.workflow_builders.ImageSaveStageBuilder import ImageSaveStageBuilder
from modules.workflow_builders.ModelApplyStageBuilder import ModelApplyStageBuilder
from modules.workflow_builders.PoseApplyStageBuilder import PoseApplyStageBuilder
from modules.workflow_builders.UpscaleImageStageBuilder import UpscaleImageStageBuilder
from modules.types import (FaceInput,
                           ImageStageOutput,
                           InpaintInput, ModelApplyStageOutput,
                           PoseInput,
                           ImageGenerationInput,
                           UserInput
                           )
# # autopep8: off
from comfy_script.runtime import Workflow, load
load()
from comfy_script.runtime.nodes import (LoadImageMask, 
                                        LoadImageFromPath, 
                                        InvertMask, 
                                        ImageResize, 
                                        VAEEncodeForInpaint, 
                                        SaveImage, 
                                        ControlNetLoader, 
                                        ControlNets, 
                                        ControlNetApply)
# # autopep8: on


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

        dw_img, dw_kp = Preprocessor(
            image=input_image, detect_face=True, detect_body=True, detect_hand=False, resolution=512)

        control_net_model = ControlNetLoader(
            control_net_name=ControlNets.control_v11p_sd15_openpose_fp16)

        conditioning = ControlNetApply(
            conditioning=models.positive,
            control_net=control_net_model,
            image=dw_img,
            strength=1.0)

        models = ModelApplyStageOutput(model=models.model, clip=models.clip,
                                       vae=models.vae, positive=conditioning, negative=models.negative)

        latent = VAEEncodeForInpaint(pixels=resized_image,
                                     mask=resized_mask,
                                     vae=models.vae,
                                     grow_mask_by=20)

        image_output = image_generation.inpaint(model_input=model_input,
                                                models=models,
                                                latent_input=latent)

        image_batch = SaveImage(
            images=image_output.image, filename_prefix='test')

    # print(wf.api_format_json())
    result_image = await image_batch.wait().get(0)

    return result_image

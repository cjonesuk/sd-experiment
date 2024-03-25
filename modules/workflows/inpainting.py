
from strenum import StrEnum
from modules.workflow_builders.ImageGenerationStageBuilder import ImageGenerationStageBuilder
from modules.workflow_builders.ImageSaveStageBuilder import ImageSaveStageBuilder
from modules.workflow_builders.ModelApplyStageBuilder import ModelApplyStageBuilder
from modules.workflow_builders.PoseApplyStageBuilder import PoseApplyStageBuilder
from modules.workflow_builders.UpscaleImageStageBuilder import UpscaleImageStageBuilder
from modules.types import (ImageInput, InpaintInput,
                           ModelApplyStageOutput,
                           PoseEstimationInput,
                           ImageGenerationInput,
                           UserInput,
                           seed)

# # autopep8: off
from comfy_script.runtime import Workflow, load
load()
from comfy_script.runtime.nodes import (LoadImageMask, 
                                        LoadImageFromPath, 
                                        InvertMask, 
                                        ImageResize, 
                                        PreviewImage,
                                        VAEEncodeForInpaint, 
                                        ControlNetLoader, 
                                        ControlNets, 
                                        ControlNetApply,
                                        DWPreprocessor,
                                        ImageCompositeMasked,
                                        GrowMask,
                                        InpaintModelConditioning,
                                        UpscaleModelLoader,
                                        UltimateSDUpscale,
                                        Samplers,
                                        Schedulers, 
                                        RecommendedResCalc)
# # autopep8: on


model_apply = ModelApplyStageBuilder()
poses = PoseApplyStageBuilder()
image_generation = ImageGenerationStageBuilder()
upscaled_image_generation = UpscaleImageStageBuilder()
image_saving = ImageSaveStageBuilder()


class InpaintStages(StrEnum):
    INPAINT = 'inpaint'
    UPSCALE = 'upscale'


async def run_inpaint_workflow(
        stage: InpaintStages,
        user_input: UserInput,
        model_input: ImageGenerationInput,
        pose_estimation_input: PoseEstimationInput,
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

        bigger_mask = GrowMask(mask=resized_mask,
                               expand=12,
                               tapered_corners=True)

        dw_img, _ = DWPreprocessor(image=input_image,
                                   detect_face=pose_estimation_input.detect_face,
                                   detect_body=pose_estimation_input.detect_body,
                                   detect_hand=pose_estimation_input.detect_hands,
                                   resolution=512)

        control_net_model = ControlNetLoader(
            control_net_name=ControlNets.control_v11p_sd15_openpose_fp16)

        conditioning = ControlNetApply(conditioning=models.positive,
                                       control_net=control_net_model,
                                       image=dw_img,
                                       strength=0.5)

        models = ModelApplyStageOutput(model=models.model,
                                       clip=models.clip,
                                       vae=models.vae,
                                       positive=conditioning,
                                       negative=models.negative)

        positive, negative, latent = InpaintModelConditioning(positive=models.positive,
                                                              negative=models.negative,
                                                              vae=models.vae,
                                                              pixels=resized_image,
                                                              mask=bigger_mask)

        models = ModelApplyStageOutput(model=models.model,
                                       clip=models.clip,
                                       vae=models.vae,
                                       positive=positive,
                                       negative=negative)

        image_output = image_generation.inpaint(model_input=model_input,
                                                models=models,
                                                latent_input=latent)

        composited = image_output.image
        composited = ImageCompositeMasked(destination=input_image,
                                          source=image_output.image,
                                          x=0,
                                          y=0,
                                          resize_source=True,
                                          mask=bigger_mask)

        image_preview = PreviewImage(images=composited)

    result = wf.task.wait_result(image_preview)
    result_image = await result.get(0)

    print(wf.api_format_json())

    return result_image


async def run_inpaint_upscale_workflow(
        stage: InpaintStages,
        user_input: UserInput,
        model_input: ImageGenerationInput,
        image_input: ImageInput,
):
    print('run_inpaint_workflow at stage:', stage)

    with Workflow(wait=True, cancel_all=True) as wf:
        models = model_apply.load(
            user_input,
            model_input)

        input_image, _ = LoadImageFromPath(image=image_input.image_path)

        upscale_model = UpscaleModelLoader(model_name='RealESRGAN_x4plus.pth')

        # des = RecommendedResCalc(desiredXSIZE=)

        upscaled_image = UltimateSDUpscale(image=input_image,
                                           model=models.model,
                                           positive=models.positive,
                                           negative=models.negative,
                                           vae=models.vae,
                                           upscale_by=1.5,
                                           seed=seed,
                                           steps=40,
                                           cfg=8.5,
                                           sampler_name=Samplers.dpmpp_2m_sde_gpu,
                                           scheduler=Schedulers.karras,
                                           denoise=0.15,
                                           upscale_model=upscale_model,
                                           tile_width=768,
                                           tile_height=768,)

        upscaled_image_preview = PreviewImage(images=upscaled_image)

    result = wf.task.wait_result(upscaled_image_preview)
    result_image = await result.get(0)

    # print(wf.api_format_json())

    return result_image

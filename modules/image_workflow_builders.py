# autopep8: off
from strenum import StrEnum
from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
# autopep8: on


seed = 156680208700281


realErsgan_x4plus_model_path = 'RealERSGAN_x4plus.pth'
universalUpscaler4x = '4x_UniversalUpscalerV2-Sharp_101000_G.pth'


class IPAdapters(StrEnum):
    IP_ADAPTER_SD15 = 'sd15\\ip-adapter_sd15.safetensors'
    IP_ADAPTER_SD15_LIGHT = 'sd15\\ip-adapter_sd15_light.safetensors'
    IP_ADAPTER_PLUS_SD15 = 'sd15\\ip-adapter-plus_sd15.safetensors'
    IP_ADAPTER_PLUS_FACE_SD15 = 'sd15\\ip-adapter-plus-face_sd15.safetensors'
    IP_ADAPTER_FULL_FACE_SD15 = 'sd15\\ip-adapter-full-face_sd15.safetensors'
    IP_ADAPTER_SD15_VIT_G = 'sd15\\ip-adapter_sd15_vit-G.safetensors'


class FaceIDAdapters(StrEnum):
    IP_ADAPTER_FACEID_SD15 = 'sd15\\faceid\\ip-adapter-faceid_sd15.bin'
    IP_ADAPTER_FACEID_PLUS_SD15 = 'sd15\\faceid\\ip-adapter-faceid-plus_sd15.bin'
    IP_ADAPTER_FACEID_PLUSV2_SD15 = 'sd15\\faceid\\ip-adapter-faceid-plusv2_sd15.bin'
    IP_ADAPTER_FACEID_PORTRAIT_SD15 = 'sd15\\faceid\\ip-adapter-faceid-portrait_sd15.bin'


class FaceIDLoras(StrEnum):
    FACEID = 'sd15\\faceid\\ip-adapter-faceid_sd15_lora.safetensors'
    FACEID_PLUS = 'sd15\\faceid\\ip-adapter-faceid-plus_sd15_lora.safetensors'
    FACEID_PLUSV2 = 'sd15\\faceid\\ip-adapter-faceid-plusv2_sd15_lora.safetensors'


ip_adapter_to_clip_vision = {
    IPAdapters.IP_ADAPTER_SD15: CLIPVisions.CLIP_ViT_H_14_laion2B_s32B_b79K,
    IPAdapters.IP_ADAPTER_SD15_LIGHT: CLIPVisions.CLIP_ViT_H_14_laion2B_s32B_b79K,
    IPAdapters.IP_ADAPTER_PLUS_SD15: CLIPVisions.CLIP_ViT_H_14_laion2B_s32B_b79K,
    IPAdapters.IP_ADAPTER_PLUS_FACE_SD15: CLIPVisions.CLIP_ViT_H_14_laion2B_s32B_b79K,
    IPAdapters.IP_ADAPTER_FULL_FACE_SD15: CLIPVisions.CLIP_ViT_H_14_laion2B_s32B_b79K,
    IPAdapters.IP_ADAPTER_SD15_VIT_G: CLIPVisions.CLIP_ViT_bigG_14_laion2B_39B_b160k,
}

faceid_to_clip_vision = {
    FaceIDAdapters.IP_ADAPTER_FACEID_SD15: None,
    FaceIDAdapters.IP_ADAPTER_FACEID_PLUS_SD15: CLIPVisions.CLIP_ViT_H_14_laion2B_s32B_b79K,
    FaceIDAdapters.IP_ADAPTER_FACEID_PLUSV2_SD15: CLIPVisions.CLIP_ViT_H_14_laion2B_s32B_b79K,
    FaceIDAdapters.IP_ADAPTER_FACEID_PORTRAIT_SD15: None
}

faceid_to_lora = {
    FaceIDAdapters.IP_ADAPTER_FACEID_SD15: FaceIDLoras.FACEID,
    FaceIDAdapters.IP_ADAPTER_FACEID_PLUS_SD15: FaceIDLoras.FACEID_PLUS,
    FaceIDAdapters.IP_ADAPTER_FACEID_PLUSV2_SD15: FaceIDLoras.FACEID_PLUSV2,
    FaceIDAdapters.IP_ADAPTER_FACEID_PORTRAIT_SD15: None
}


class UserInput:
    def __init__(self, prompt: str, negative_prompt: str):
        self.prompt = prompt
        self.negative_prompt = negative_prompt


class ImageGenerationInput:
    def __init__(self, checkpoint: str, num_inference_steps: int, guidance_scale: float):
        self.checkpoint = checkpoint
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale


class PoseInput:
    def __init__(self, image_path: str):
        self.image_path = image_path


class FaceInput:
    def __init__(self, image_path: str):
        self.image_path = image_path


class PreparedFaceOutput:
    def __init__(self, image: Image):
        self.image = image


class ModelApplyStageOutput:
    def __init__(self, model: Model, clip: Clip, vae: Vae, positive: CLIPTextEncode, negative: CLIPTextEncode):
        self.model = model
        self.clip = clip
        self.vae = vae
        self.positive = positive
        self.negative = negative


class ImageStageOutput:
    def __init__(self, image: VAEDecode):
        self.image = image


class UpscaleImageStageOutput:
    def __init__(self, image: VAEDecode):
        self.image = image


class ModelApplyStageBuilder:
    def apply_workflow(self, user_input: UserInput, image_generation_input: ImageGenerationInput):
        model, clip, vae = CheckpointLoaderSimple(
            ckpt_name=image_generation_input.checkpoint)
        positive_conditioning = CLIPTextEncode(user_input.prompt, clip)
        negative_conditioning = CLIPTextEncode(
            user_input.negative_prompt, clip)

        return ModelApplyStageOutput(model=model, clip=clip, vae=vae, positive=positive_conditioning, negative=negative_conditioning)


class PoseApplyStageBuilder:
    def apply_pose_conditioning(self, models: ModelApplyStageOutput, pose_input: PoseInput):
        pose_image, pose_mask = LoadImageFromPath(image=pose_input.image_path)

        control_net_model = ControlNetLoader(
            control_net_name=ControlNets.control_v11p_sd15_openpose_fp16)

        conditioning = ControlNetApply(
            conditioning=models.positive,
            control_net=control_net_model,
            image=pose_image,
            strength=1.0)

        return ModelApplyStageOutput(model=models.model, clip=models.clip, vae=models.vae, positive=conditioning, negative=models.negative)


class FacePreparationStageBuilder:
    def prepare_face(self, face_input: FaceInput):
        face_image, face_mask = LoadImageFromPath(image=face_input.image_path)

        prepared_image = PrepImageForInsightFace(
            image=face_image, crop_position='center', sharpening=0, pad_around=True)

        return PreparedFaceOutput(prepared_image)


class FaceIdApplyStageBuilder:
    def apply_face_id_conditioning(self, models: ModelApplyStageOutput, prepared_face: PreparedFaceOutput):
        faceid_model_name = FaceIDAdapters.IP_ADAPTER_FACEID_PLUSV2_SD15
        clip_vision_model_name = faceid_to_clip_vision[faceid_model_name]
        lora_model_name = faceid_to_lora[faceid_model_name]

        ipadapter = IPAdapterModelLoader(ipadapter_file=faceid_model_name)
        insightface = InsightFaceLoader(provider='CUDA')
        clip_vision = CLIPVisionLoader(clip_name=clip_vision_model_name)
        model_with_lora = LoraLoaderModelOnly(
            model=models.model, lora_name=lora_model_name, strength_model=1.0)

        model = IPAdapterApplyFaceID(ipadapter=ipadapter,
                                     clip_vision=clip_vision,
                                     insightface=insightface,
                                     image=prepared_face.image,
                                     model=model_with_lora,
                                     weight=0.4,
                                     weight_v2=1.0,
                                     faceid_v2=True,
                                     noise=0.0)

        return ModelApplyStageOutput(model=model, clip=models.clip, vae=models.vae, positive=models.positive, negative=models.negative)


class ImageGenerationStageBuilder:
    def generate_with_latent_upscale(self, model_input: ImageGenerationInput, models: ModelApplyStageOutput):
        stage_one_end = int(model_input.num_inference_steps * 0.6)
        stage_two_start = stage_one_end + 1

        empty_latent = EmptyLatentImage(512, 512, 1)

        latent = KSamplerAdvanced(model=models.model,
                                  noise_seed=seed,
                                  steps=model_input.num_inference_steps,
                                  cfg=model_input.guidance_scale,
                                  sampler_name='euler',
                                  scheduler='normal',
                                  positive=models.positive,
                                  negative=models.negative,
                                  latent_image=empty_latent,
                                  end_at_step=stage_one_end)

        PreviewImage(VAEDecode(latent, models.vae))

        upscaled_latent = LatentUpscaleBy(
            samples=latent,
            upscale_method='nearest-exact',
            scale_by=1.5)

        upscaled_latent = KSamplerAdvanced(model=models.model,
                                           noise_seed=seed,
                                           steps=model_input.num_inference_steps,
                                           cfg=model_input.guidance_scale,
                                           sampler_name='euler',
                                           scheduler='normal',
                                           positive=models.positive,
                                           negative=models.negative,
                                           latent_image=upscaled_latent,
                                           start_at_step=stage_two_start)

        output_image = VAEDecode(upscaled_latent, models.vae)
        PreviewImage(output_image)

        return ImageStageOutput(output_image)


class UpscaleImageStageBuilder:
    def upscale_simple(self, image_generation_output: ImageStageOutput):
        upscaled_image = ImageScaleBy(
            image=image_generation_output.image, upscale_method='nearest-exact', scale_by=2)

        return UpscaleImageStageOutput(upscaled_image)

    def upscale_extended(self, image_generation_output: ImageStageOutput):
        upscale_model = UpscaleModelLoader(UpscaleModels.RealESRGAN_x4plus)

        upscaled_image = ImageUpscaleWithModel(
            upscale_model=upscale_model, image=image_generation_output.image)

        return UpscaleImageStageOutput(upscaled_image)

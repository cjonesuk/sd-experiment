from strenum import StrEnum

from modules.types import ModelApplyStageOutput, PreparedFaceOutput

from comfy_script.runtime.nodes import CLIPVisionLoader, IPAdapterApplyFaceID, IPAdapterModelLoader, InsightFaceLoader, LoraLoaderModelOnly, CLIPVisions


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

# autopep8: off
from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
# autopep8: on


seed = 156680208700281


realErsgan_x4plus_model_path = 'RealERSGAN_x4plus.pth'
universalUpscaler4x = '4x_UniversalUpscalerV2-Sharp_101000_G.pth'


class UserInput:
    def __init__(self, prompt: str, negative_prompt: str):
        self.prompt = prompt
        self.negative_prompt = negative_prompt


class ImageGenerationInput:
    def __init__(self, checkpoint: str, num_inference_steps: int, guidance_scale: float):
        self.checkpoint = checkpoint
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale


class InpaintInput:
    def __init__(self, image_path: str, mask_path: str):
        self.image_path = image_path
        self.mask_path = mask_path


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

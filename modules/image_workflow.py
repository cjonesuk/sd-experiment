# autopep8: off
from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
# autopep8: on

seed = 156680208700281


class UserInput:
    def __init__(self, prompt: String, negative_prompt: String):
        self.prompt = prompt
        self.negative_prompt = negative_prompt


class ModelApplyStageInput:
    def __init__(self, model: str, num_inference_steps: int, guidance_scale: float):
        self.model = model
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale


class ModelApplyStageOutput:
    def __init__(self, model: Model, clip: Clip, vae: Vae, positive: CLIPTextEncode, negative: CLIPTextEncode, ):
        self.model = model
        self.clip = clip
        self.vae = vae
        self.positive = positive
        self.negative = negative


class ModelApplyStageBuilder:
    def apply_workflow(self, user_input: UserInput, model_input: ModelApplyStageInput):
        print('model', model_input.model)
        print('steps', model_input.num_inference_steps)
        print('guidance', model_input.guidance_scale)
        print('prompt', user_input.prompt)
        print('negative_prompt', user_input.negative_prompt)

        model, clip, vae = CheckpointLoaderSimple(model_input.model)
        positive_conditioning = CLIPTextEncode(user_input.prompt, clip)
        negative_conditioning = CLIPTextEncode(
            user_input.negative_prompt, clip)

        return ModelApplyStageOutput(model=model, clip=clip, vae=vae, positive=positive_conditioning, negative=negative_conditioning)


class ImageGenerationStageBuilder:
    def apply_workflow(self, model_input: ModelApplyStageInput, models: ModelApplyStageOutput):
        stage_one_end = int(model_input.num_inference_steps * 0.6)
        stage_two_start = stage_one_end + 1

        print('steps', model_input.num_inference_steps)
        print('stage_one_end', stage_one_end)
        print('stage_two_start', stage_two_start)

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


class ImageStageOutput:
    def __init__(self, image: VAEDecode):
        self.image = image

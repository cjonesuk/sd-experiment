from modules.types import ImageGenerationInput, ImageStageOutput, ModelApplyStageOutput, seed


from comfy_script.runtime.nodes import EmptyLatentImage, KSamplerAdvanced, Latent, LatentUpscaleBy, PreviewImage, VAEDecode


class ImageGenerationStageBuilder:
    def inpaint(self, model_input: ImageGenerationInput, models: ModelApplyStageOutput, latent_input: Latent):
        latent = KSamplerAdvanced(model=models.model,
                                  noise_seed=seed,
                                  steps=model_input.num_inference_steps,
                                  cfg=model_input.guidance_scale,
                                  sampler_name='euler',
                                  scheduler='normal',
                                  positive=models.positive,
                                  negative=models.negative,
                                  latent_image=latent_input,
                                  end_at_step=model_input.num_inference_steps)

        output_image = VAEDecode(latent, models.vae)
        PreviewImage(output_image)

        return ImageStageOutput(output_image)

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

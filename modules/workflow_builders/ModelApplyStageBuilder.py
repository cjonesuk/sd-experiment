from modules.types import ImageGenerationInput, ModelApplyStageOutput, UserInput


from comfy_script.runtime.nodes import CLIPTextEncode, CheckpointLoaderSimple


class ModelApplyStageBuilder:
    def load(self, user_input: UserInput, image_generation_input: ImageGenerationInput):
        model, clip, vae = CheckpointLoaderSimple(
            ckpt_name=image_generation_input.checkpoint)
        positive_conditioning = CLIPTextEncode(user_input.prompt, clip)
        negative_conditioning = CLIPTextEncode(
            user_input.negative_prompt, clip)

        return ModelApplyStageOutput(model=model, clip=clip, vae=vae, positive=positive_conditioning, negative=negative_conditioning)

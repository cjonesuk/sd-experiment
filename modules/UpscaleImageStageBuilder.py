from modules.types import ImageStageOutput, UpscaleImageStageOutput


from comfy_script.runtime.nodes import ImageScaleBy, ImageUpscaleWithModel, UpscaleModelLoader, UpscaleModels


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

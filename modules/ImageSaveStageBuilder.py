from modules.types import ImageStageOutput


from comfy_script.runtime.nodes import SaveImage


class ImageSaveStageBuilder:
    def save_image(self, image_generation_output: ImageStageOutput, filename_prefix: str):
        return SaveImage(images=image_generation_output.image, filename_prefix=filename_prefix)

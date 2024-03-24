from strenum import StrEnum
from modules.FaceIdApplyStageBuilder import FaceIdApplyStageBuilder
from modules.FacePreparationStageBuilder import FacePreparationStageBuilder
from modules.ImageGenerationStageBuilder import ImageGenerationStageBuilder
from modules.ImageSaveStageBuilder import ImageSaveStageBuilder
from modules.ModelApplyStageBuilder import ModelApplyStageBuilder
from modules.PoseApplyStageBuilder import PoseApplyStageBuilder
from modules.UpscaleImageStageBuilder import UpscaleImageStageBuilder
from modules.types import FaceInput, ImageStageOutput, PoseInput, ImageGenerationInput, UserInput

# autopep8: off
from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
# autopep8: on

model_apply = ModelApplyStageBuilder()
poses = PoseApplyStageBuilder()
face_preparation = FacePreparationStageBuilder()
faces = FaceIdApplyStageBuilder()
image_generation = ImageGenerationStageBuilder()
upscaled_image_generation = UpscaleImageStageBuilder()
image_saving = ImageSaveStageBuilder()


class GenerateImageWithPoseWorkflowStages(StrEnum):
    GENERATE_IMAGE = 'generate_image'
    UPSCALE_IMAGE = 'upscale_image'


async def run_generate_image_with_pose_workflow(
        stage: GenerateImageWithPoseWorkflowStages,
        user_input: UserInput,
        model_input: ImageGenerationInput,
        pose_input: PoseInput,
        face_input: FaceInput,
):
    print('run_generate_image_with_pose_workflow at stage:', stage)

    with Workflow(wait=True, cancel_all=True):
        models = model_apply.load(
            user_input,
            model_input)

        prepared_face = face_preparation.prepare_face(face_input)

        models = poses.apply_pose_conditioning(models, pose_input)
        models = faces.apply_face_id_conditioning(models, prepared_face)

        output = image_generation.generate_with_latent_upscale(
            model_input,
            models)

        image_batch = image_saving.save_image(output, 'PY_ComfyUI')

        if (stage == GenerateImageWithPoseWorkflowStages.UPSCALE_IMAGE):
            upscaled = upscaled_image_generation.upscale_simple(output)
            image_batch = image_saving.save_image(upscaled, 'PY_ComfyUI')

    result_image = await image_batch.wait().get(0)

    return result_image

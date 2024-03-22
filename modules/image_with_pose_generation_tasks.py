from modules.image_workflow_builders import FaceIdApplyStageBuilder, FaceInput, FacePreparationStageBuilder, ImageGenerationStageBuilder, ModelApplyStageBuilder, PoseApplyStageBuilder, PoseInput, UpscaleImageStageBuilder, ModelApplyStageInput, UserInput

# autopep8: off
from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
# autopep8: on

analogMadness = "sd15\\analogMadness_v70.safetensors"

model_apply = ModelApplyStageBuilder()
poses = PoseApplyStageBuilder()
face_preparation = FacePreparationStageBuilder()
faces = FaceIdApplyStageBuilder()
image_generation = ImageGenerationStageBuilder()
upscaled_image_generation = UpscaleImageStageBuilder()


async def run_generate_image_with_pose_workflow(prompt, negative_prompt, num_inference_steps, guidance_scale, pose_image: Image, face_image: Image):
    user_input = UserInput(prompt, negative_prompt)

    model_input = ModelApplyStageInput(
        analogMadness,
        num_inference_steps,
        guidance_scale)

    pose_input = PoseInput(pose_image)
    face_input = FaceInput(face_image)

    image_batch = None

    with Workflow(wait=True, cancel_all=True):
        models = model_apply.apply_workflow(
            user_input,
            model_input)

        prepared_face = face_preparation.prepare_face(face_input)

        models = poses.apply_pose_conditioning(models, pose_input)
        models = faces.apply_face_id_conditioning(models, prepared_face)

        output = image_generation.generate_with_latent_upscale(
            model_input,
            models)

        image_batch = SaveImage(output.image, 'PY_ComfyUI')

    result_image = await image_batch.wait().get(0)

    return result_image

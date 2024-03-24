from modules.types import ModelApplyStageOutput, PoseInput


from comfy_script.runtime.nodes import ControlNetApply, ControlNetLoader, ControlNets, LoadImageFromPath


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

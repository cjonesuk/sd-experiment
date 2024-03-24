from modules.types import FaceInput, PreparedFaceOutput


from comfy_script.runtime.nodes import LoadImageFromPath, PrepImageForInsightFace


class FacePreparationStageBuilder:
    def prepare_face(self, face_input: FaceInput):
        face_image, face_mask = LoadImageFromPath(image=face_input.image_path)

        prepared_image = PrepImageForInsightFace(
            image=face_image, crop_position='center', sharpening=0, pad_around=True)

        return PreparedFaceOutput(prepared_image)

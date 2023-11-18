from typing import Any, Dict, Set, ClassVar, List

from pydantic import (
    confloat,
    BaseModel,
    Field,
    validator,
)

from .api import alwayson_scripts_pyload_wrapper

ad_models = {
    "mediapipe_face_short",
    "mediapipe_face_full",
    "mediapipe_face_mesh",
    "mediapipe_face_mesh_eyes_only",
    "face_yolov8n.pt",
    "face_yolov8s.pt",
    "hand_yolov8n.pt",
    "person_yolov8n-seg.pt",
    "person_yolov8s-seg.pt",
}


class ADetailerArgs(BaseModel):
    """
    api details see https://github.com/Bing-su/adetailer/wiki/API
    """

    class Config:
        allow_mutation = False
        validate_assignment = True

    ad_models: ClassVar[Set[str]] = ad_models

    ad_model: str = Field(default="mediapipe_face_full")
    ad_prompt: str = ""
    ad_negative_prompt: str = ""
    ad_denoising_strength: confloat(ge=0.0, le=1.0) = 0.4

    @alwayson_scripts_pyload_wrapper("ADetailer")
    def make_pyload(self) -> List[Dict[str, Any]]:
        return [True, False, self.dict()]

    @validator("ad_model")
    def validate_ad_model(cls, v):
        if v not in ad_models:
            raise ValueError(f"ad_model must be one of {ad_models}")
        return v

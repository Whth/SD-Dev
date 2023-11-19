from enum import Enum
from typing import Any, Dict, Set, ClassVar, List

from pydantic import (
    confloat,
    BaseModel,
    Field,
    validator,
)

from .api import alwayson_scripts_pyload_wrapper


class ModelType(Enum):
    MEDIAPIPE_FACE_SHORT = "mediapipe_face_short"
    MEDIAPIPE_FACE_FULL = "mediapipe_face_full"
    MEDIAPIPE_FACE_MESH = "mediapipe_face_mesh"
    MEDIAPIPE_FACE_MESH_EYES_ONLY = "mediapipe_face_mesh_eyes_only"
    FACE_YOLOV8N = "face_yolov8n.pt"
    FACE_YOLOV8S = "face_yolov8s.pt"
    HAND_YOLOV8N = "hand_yolov8n.pt"
    PERSON_YOLOV8NSEG = "person_yolov8n-seg.pt"
    PERSON_YOLOV8SSEG = "person_yolov8s-seg.pt"


ad_models: Set[str] = {
    ModelType.MEDIAPIPE_FACE_SHORT.value,
    ModelType.MEDIAPIPE_FACE_FULL.value,
    ModelType.MEDIAPIPE_FACE_MESH.value,
    ModelType.MEDIAPIPE_FACE_MESH_EYES_ONLY.value,
    ModelType.FACE_YOLOV8N.value,
    ModelType.FACE_YOLOV8S.value,
    ModelType.HAND_YOLOV8N.value,
    ModelType.PERSON_YOLOV8NSEG.value,
    ModelType.PERSON_YOLOV8SSEG.value,
}


class ADetailerUnit(BaseModel):
    class Config:
        allow_mutation = False
        validate_assignment = True

    ad_models: ClassVar[Set[str]] = ad_models

    ad_model: str = Field(default=ModelType.MEDIAPIPE_FACE_FULL.value)
    ad_prompt: str = ""
    ad_negative_prompt: str = ""
    ad_denoising_strength: confloat(ge=0.0, le=1.0) = 0.4
    ad_controlnet_model: str = None
    ad_controlnet_module: str = None
    ad_controlnet_weight: confloat(ge=0.0, le=1.0) = 1.0
    ad_controlnet_guidance_start: confloat(ge=0.0, le=1.0) = 0.1
    ad_controlnet_guidance_end: confloat(ge=0.0, le=1.0) = 0.9
    ad_use_clip_skip: bool = False
    ad_clip_skip: int = 2

    @validator("ad_model")
    def validate_ad_model(cls, v):
        if v not in cls.ad_models:
            raise ValueError(f"ad_model must be one of {cls.ad_models}")
        return v


class ADetailerArgs(BaseModel):
    """
    api details see https://github.com/Bing-su/adetailer/wiki/API
    """

    class Config:
        allow_mutation = False
        validate_assignment = True

    ad_unit: List[ADetailerUnit] = Field(default_factory=list)

    @alwayson_scripts_pyload_wrapper("ADetailer")
    def make_pyload(self) -> List[Dict[str, Any]]:
        return [unit.dict(exclude_none=True) for unit in self.ad_unit]

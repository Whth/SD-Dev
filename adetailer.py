from typing import Any, Dict

from pydantic import (
    confloat,
    BaseModel,
)


class ADetailerArgs(BaseModel):
    ad_model: str = "mediapipe_face_full.pt"
    ad_prompt: str = ""
    ad_negative_prompt: str = ""
    ad_denoising_strength: confloat(ge=0.0, le=1.0) = 0.4

    def make_pyload(self) -> Dict[str, Any]:
        return {"ADetailer": {"args": [self.dict()]}}

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VideoPipeline(str, Enum):
    DISTILLED = "distilled"
    DEV = "dev"
    DEV_TWO_STAGE = "dev-two-stage"
    DEV_TWO_STAGE_HQ = "dev-two-stage-hq"


class VideoResponseFormat(str, Enum):
    URL = "url"
    B64_JSON = "b64_json"


class VideoGenerationRequest(BaseModel):
    prompt: str = Field(..., max_length=4000)
    model: str = "Lightricks/LTX-2"
    width: int = Field(default=512, ge=64)
    height: int = Field(default=512, ge=64)
    num_frames: int = Field(default=97, ge=1, le=257)
    fps: int = Field(default=24, ge=1, le=60)
    seed: Optional[int] = None
    pipeline: VideoPipeline = VideoPipeline.DISTILLED
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = None
    cfg_scale: float = Field(default=3.0, ge=0.0, le=20.0)
    response_format: VideoResponseFormat = VideoResponseFormat.URL
    image: Optional[str] = None
    image_url: Optional[str] = None
    image_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    tiling: str = "auto"

    model_config = {"extra": "allow"}

    def get_extra_params(self) -> Dict[str, Any]:
        standard = {
            "prompt", "model", "width", "height", "num_frames", "fps",
            "seed", "pipeline", "negative_prompt", "num_inference_steps",
            "cfg_scale", "response_format", "image", "image_url",
            "image_strength", "tiling",
        }
        return {k: v for k, v in self.model_dump().items() if k not in standard}


class VideoObject(BaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None


class VideoGenerationResponse(BaseModel):
    created: int
    data: List[VideoObject]

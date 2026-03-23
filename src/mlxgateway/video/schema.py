from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class VideoPipeline(str, Enum):
    DISTILLED = "distilled"
    DEV = "dev"
    DEV_TWO_STAGE = "dev-two-stage"
    DEV_TWO_STAGE_HQ = "dev-two-stage-hq"


class VideoTiling(str, Enum):
    AUTO = "auto"
    NONE = "none"
    CONSERVATIVE = "conservative"
    DEFAULT = "default"
    AGGRESSIVE = "aggressive"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"


class VideoResponseFormat(str, Enum):
    URL = "url"
    B64_JSON = "b64_json"


class VideoGenerationRequest(BaseModel):
    prompt: str = Field(..., max_length=4000)
    model: str = "prince-canuma/LTX-2.3-distilled"
    text_encoder_repo: Optional[str] = Field(
        default="mlx-community/gemma-3-12b-it-bf16",
        description="HuggingFace repo for the text encoder. "
        "Set to null for models that bundle their own text encoder (e.g. Lightricks/LTX-2).",
    )
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
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
    tiling: VideoTiling = VideoTiling.AUTO

    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def validate_constraints(self):
        if self.width % 32 != 0:
            raise ValueError(f"width must be divisible by 32, got {self.width}")
        if self.height % 32 != 0:
            raise ValueError(f"height must be divisible by 32, got {self.height}")
        if self.num_frames > 1 and (self.num_frames - 1) % 8 != 0:
            raise ValueError(
                f"num_frames must be 1 + 8*k (e.g. 1, 9, 17, 25, ..., 257), got {self.num_frames}"
            )
        if self.image and self.image_url:
            raise ValueError("Provide either 'image' (base64) or 'image_url', not both")
        return self

    def get_extra_params(self) -> Dict[str, Any]:
        standard = {
            "prompt", "model", "text_encoder_repo", "width", "height",
            "num_frames", "fps", "seed", "pipeline", "negative_prompt",
            "num_inference_steps", "cfg_scale", "response_format", "image",
            "image_url", "image_strength", "tiling",
        }
        return {k: v for k, v in self.model_dump().items() if k not in standard}


class VideoObject(BaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None


class VideoGenerationResponse(BaseModel):
    created: int
    data: List[VideoObject]

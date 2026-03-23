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
    image: Optional[str] = Field(
        default=None,
        description="Base64-encoded first frame image for I2V. "
        "Conditions the video to start from this image.",
    )
    image_url: Optional[str] = Field(
        default=None,
        description="URL of the first frame image for I2V.",
    )
    end_image: Optional[str] = Field(
        default=None,
        description="Base64-encoded last frame image for I2V. "
        "Conditions the video to end at this image. "
        "Can be used alone or together with 'image' for dual-frame conditioning.",
    )
    end_image_url: Optional[str] = Field(
        default=None,
        description="URL of the last frame image for I2V.",
    )
    image_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    image_frame_idx: int = Field(
        default=0,
        description="Frame index to condition the image on. "
        "0 = first frame (default). Ignored when end_image is provided "
        "(automatically set to 0 for image and -1 for end_image).",
    )
    audio: bool = Field(
        default=True,
        description="Generate synchronized audio alongside the video. "
        "LTX-2 was jointly trained on audio+video; the audio is embedded in the output mp4. "
        "Automatically disabled when audio_file/audio_file_url is provided (A2V mode).",
    )
    audio_cfg_scale: float = Field(
        default=7.0, ge=0.0, le=20.0,
        description="CFG guidance scale for audio generation.",
    )
    audio_file: Optional[str] = Field(
        default=None,
        description="Base64-encoded audio file for A2V (Audio-to-Video). "
        "The video will be driven by this audio's rhythm and content.",
    )
    audio_file_url: Optional[str] = Field(
        default=None,
        description="URL of audio file for A2V.",
    )
    audio_start_time: float = Field(
        default=0.0, ge=0.0,
        description="Start time offset (seconds) to begin reading the A2V audio file.",
    )
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
        if self.end_image and self.end_image_url:
            raise ValueError("Provide either 'end_image' (base64) or 'end_image_url', not both")
        has_first = bool(self.image or self.image_url)
        has_last = bool(self.end_image or self.end_image_url)
        if has_first and has_last and self.num_frames < 9:
            raise ValueError(
                "Dual-frame conditioning (image + end_image) requires at least 9 frames"
            )
        if self.audio_file and self.audio_file_url:
            raise ValueError("Provide either 'audio_file' (base64) or 'audio_file_url', not both")
        # A2V and audio generation are mutually exclusive. Auto-disable audio
        # generation when A2V input is provided (user didn't explicitly set audio).
        if self.audio_file or self.audio_file_url:
            self.audio = False
        return self

    def get_extra_params(self) -> Dict[str, Any]:
        standard = {
            "prompt", "model", "text_encoder_repo", "width", "height",
            "num_frames", "fps", "seed", "pipeline", "negative_prompt",
            "num_inference_steps", "cfg_scale", "response_format", "image",
            "image_url", "end_image", "end_image_url",
            "image_strength", "image_frame_idx", "audio", "audio_cfg_scale",
            "audio_file", "audio_file_url", "audio_start_time", "tiling",
        }
        return {k: v for k, v in self.model_dump().items() if k not in standard}


class VideoObject(BaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None


class VideoGenerationResponse(BaseModel):
    created: int
    data: List[VideoObject]

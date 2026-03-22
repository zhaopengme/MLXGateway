import base64
import ipaddress
import random
import socket
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

from ..utils.logger import logger
from .schema import VideoGenerationRequest, VideoObject, VideoPipeline, VideoResponseFormat

_VIDEO_OUTPUT_DIR = Path(tempfile.gettempdir()) / "mlxgateway" / "videos"
_VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _pipeline_enum(pipeline: VideoPipeline):
    """Convert our schema enum to mlx_video's PipelineType."""
    from mlx_video.models.ltx_2.generate import PipelineType
    return {
        VideoPipeline.DISTILLED: PipelineType.DISTILLED,
        VideoPipeline.DEV: PipelineType.DEV,
        VideoPipeline.DEV_TWO_STAGE: PipelineType.DEV_TWO_STAGE,
        VideoPipeline.DEV_TWO_STAGE_HQ: PipelineType.DEV_TWO_STAGE_HQ,
    }[pipeline]


def _validate_url(url: str) -> None:
    """Reject non-HTTP schemes and private/internal IP addresses (SSRF protection)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Invalid URL: no hostname")
    try:
        ip = ipaddress.ip_address(socket.gethostbyname(hostname))
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            raise ValueError("URL points to a private/internal address")
    except socket.gaierror:
        raise ValueError(f"Cannot resolve hostname: {hostname}")


def resolve_image(request: VideoGenerationRequest) -> str | None:
    """Resolve image input to a local temp file path.

    This function may perform network I/O (downloading from URL) and should
    NOT be called on the MLX worker thread. Call it from the async layer.
    """
    if request.image:
        try:
            img_data = base64.b64decode(request.image)
            tmp = _VIDEO_OUTPUT_DIR / f"i2v_input_{uuid.uuid4().hex}.png"
            tmp.write_bytes(img_data)
            return str(tmp)
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            raise ValueError("Invalid base64 image data") from e

    if request.image_url:
        _validate_url(request.image_url)
        import urllib.request
        tmp = _VIDEO_OUTPUT_DIR / f"i2v_input_{uuid.uuid4().hex}.png"
        try:
            urllib.request.urlretrieve(request.image_url, str(tmp))
            return str(tmp)
        except Exception as e:
            tmp.unlink(missing_ok=True)
            logger.error(f"Failed to download image from URL: {e}")
            raise ValueError(f"Failed to download image: {e}") from e

    return None


class VideoService:
    """Synchronous video generation service. All methods run on the MLX worker thread."""

    def generate_video(
        self,
        request: VideoGenerationRequest,
        base_url: str = "",
        image_path: str | None = None,
    ) -> VideoObject:
        from mlx_video.models.ltx_2.generate import generate_video

        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        output_filename = f"video_{uuid.uuid4().hex}.mp4"
        output_path = str(_VIDEO_OUTPUT_DIR / output_filename)

        is_i2v = image_path is not None
        mode = "I2V" if is_i2v else "T2V"

        logger.info(
            f"[{mode}] Generating video: model={request.model}, "
            f"{request.width}x{request.height}, frames={request.num_frames}, "
            f"pipeline={request.pipeline.value}, seed={seed}"
        )

        extra = request.get_extra_params()

        # Distilled uses fixed sigma schedules (steps param is ignored internally),
        # dev/dev-two-stage need explicit step counts.
        steps = request.num_inference_steps
        if steps is None:
            steps = 40 if request.pipeline == VideoPipeline.DISTILLED else 30

        gen_kwargs = {
            "model_repo": request.model,
            "prompt": request.prompt,
            "pipeline": _pipeline_enum(request.pipeline),
            "height": request.height,
            "width": request.width,
            "num_frames": request.num_frames,
            "num_inference_steps": steps,
            "cfg_scale": request.cfg_scale,
            "seed": seed,
            "fps": request.fps,
            "output_path": output_path,
            "save_frames": False,
            "verbose": True,
            "tiling": request.tiling.value,
            "stream": False,
        }

        text_encoder = extra.get("text_encoder_repo")
        if text_encoder:
            gen_kwargs["text_encoder_repo"] = text_encoder

        if request.negative_prompt is not None:
            gen_kwargs["negative_prompt"] = request.negative_prompt

        if is_i2v:
            gen_kwargs["image"] = image_path
            gen_kwargs["image_strength"] = request.image_strength

        for key in ("lora_path", "lora_strength", "enhance_prompt", "spatial_upscaler"):
            if key in extra:
                gen_kwargs[key] = extra[key]

        t0 = time.perf_counter()
        try:
            generate_video(**gen_kwargs)
        finally:
            if image_path:
                Path(image_path).unlink(missing_ok=True)
        elapsed = time.perf_counter() - t0

        logger.info(f"[{mode}] Video generated in {elapsed:.1f}s: {output_path}")

        if request.response_format == VideoResponseFormat.B64_JSON:
            video_bytes = Path(output_path).read_bytes()
            b64 = base64.b64encode(video_bytes).decode()
            Path(output_path).unlink(missing_ok=True)
            logger.info(f"[{mode}] Encoded video: {len(b64)} chars base64")
            return VideoObject(b64_json=b64, revised_prompt=request.prompt)

        url = f"{base_url}/v1/videos/files/{output_filename}" if base_url else f"file://{output_path}"
        return VideoObject(url=url, revised_prompt=request.prompt)

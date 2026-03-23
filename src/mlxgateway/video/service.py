import base64
import ipaddress
import random
import socket
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

from ..utils.logger import logger
from ..utils.static import TEMP_DIR, VIDEOS_DIR
from .schema import VideoGenerationRequest, VideoObject, VideoPipeline, VideoResponseFormat

_VIDEO_OUTPUT_DIR = VIDEOS_DIR


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
    """Reject non-HTTP schemes and private/internal IP addresses (SSRF protection).

    Note: DNS rebinding can bypass this check (TOCTOU between gethostbyname
    and urlretrieve). Acceptable for a local-network MLX gateway.
    """
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


def _resolve_single_image(b64_data: str | None, url: str | None, label: str) -> str | None:
    """Resolve a single image (base64 or URL) to a local temp file path."""
    if b64_data:
        try:
            img_data = base64.b64decode(b64_data)
            tmp = TEMP_DIR / f"i2v_{label}_{uuid.uuid4().hex}.png"
            tmp.write_bytes(img_data)
            return str(tmp)
        except Exception as e:
            logger.error(f"Failed to decode base64 {label} image: {e}")
            raise ValueError(f"Invalid base64 {label} image data") from e

    if url:
        _validate_url(url)
        import urllib.request
        tmp = TEMP_DIR / f"i2v_{label}_{uuid.uuid4().hex}.png"
        try:
            urllib.request.urlretrieve(url, str(tmp))
            return str(tmp)
        except Exception as e:
            tmp.unlink(missing_ok=True)
            logger.error(f"Failed to download {label} image from URL: {e}")
            raise ValueError(f"Failed to download {label} image: {e}") from e

    return None


def resolve_images(request: VideoGenerationRequest) -> tuple[str | None, str | None]:
    """Resolve first-frame and last-frame images to local temp file paths.

    This function may perform network I/O and should NOT be called on the
    MLX worker thread. Call it from the async layer.

    Returns (first_image_path, last_image_path).
    """
    first = _resolve_single_image(request.image, request.image_url, "first")
    last = _resolve_single_image(request.end_image, request.end_image_url, "last")
    return first, last


class VideoService:
    """Synchronous video generation service. All methods run on the MLX worker thread."""

    def generate_video(
        self,
        request: VideoGenerationRequest,
        base_url: str = "",
        first_image_path: str | None = None,
        last_image_path: str | None = None,
    ) -> VideoObject:
        from mlx_video.models.ltx_2.generate import generate_video

        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        output_filename = f"video_{uuid.uuid4().hex}.mp4"
        output_path = str(_VIDEO_OUTPUT_DIR / output_filename)

        has_first = first_image_path is not None
        has_last = last_image_path is not None
        is_i2v = has_first or has_last

        if has_first and has_last:
            mode = "I2V(first+last)"
        elif has_first:
            mode = "I2V(first)"
        elif has_last:
            mode = "I2V(last)"
        else:
            mode = "T2V"

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

        # text_encoder_repo is a required param in mlx-video's generate_video;
        # passing None tells it to use the model_repo path for the text encoder.
        gen_kwargs = {
            "model_repo": request.model,
            "text_encoder_repo": request.text_encoder_repo,
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

        if request.negative_prompt is not None:
            gen_kwargs["negative_prompt"] = request.negative_prompt

        if is_i2v:
            # mlx-video's generate_video currently supports one conditioning image.
            # When both first and last are provided, we use the first frame image
            # and log a warning about the limitation.
            if has_first and has_last:
                logger.warning(
                    "Both first and last frame images provided, but mlx-video only "
                    "supports single-image conditioning. Using first frame image. "
                    "Last frame image will be ignored until multi-conditioning is supported."
                )
                gen_kwargs["image"] = first_image_path
                gen_kwargs["image_frame_idx"] = 0
            elif has_first:
                gen_kwargs["image"] = first_image_path
                gen_kwargs["image_frame_idx"] = request.image_frame_idx
            else:
                gen_kwargs["image"] = last_image_path
                gen_kwargs["image_frame_idx"] = -1
            gen_kwargs["image_strength"] = request.image_strength

        for key in ("lora_path", "lora_strength", "enhance_prompt", "spatial_upscaler"):
            if key in extra:
                gen_kwargs[key] = extra[key]

        t0 = time.perf_counter()
        try:
            generate_video(**gen_kwargs)
        finally:
            if first_image_path:
                Path(first_image_path).unlink(missing_ok=True)
            if last_image_path:
                Path(last_image_path).unlink(missing_ok=True)
        elapsed = time.perf_counter() - t0

        logger.info(f"[{mode}] Video generated in {elapsed:.1f}s: {output_path}")

        if request.response_format == VideoResponseFormat.B64_JSON:
            video_bytes = Path(output_path).read_bytes()
            b64 = base64.b64encode(video_bytes).decode()
            Path(output_path).unlink(missing_ok=True)
            logger.info(f"[{mode}] Encoded video: {len(b64)} chars base64")
            return VideoObject(b64_json=b64, revised_prompt=request.prompt)

        url = f"{base_url}/static/videos/{output_filename}" if base_url else f"file://{output_path}"
        return VideoObject(url=url, revised_prompt=request.prompt)

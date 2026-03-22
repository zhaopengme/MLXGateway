import base64
import random
import tempfile
import time
from pathlib import Path

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


def _resolve_image(request: VideoGenerationRequest) -> str | None:
    """Resolve image input: base64 data or URL to a local temp file path."""
    if request.image:
        try:
            img_data = base64.b64decode(request.image)
            tmp = _VIDEO_OUTPUT_DIR / f"i2v_input_{int(time.time())}.png"
            tmp.write_bytes(img_data)
            return str(tmp)
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            raise ValueError("Invalid base64 image data") from e

    if request.image_url:
        import urllib.request
        tmp = _VIDEO_OUTPUT_DIR / f"i2v_input_{int(time.time())}.png"
        try:
            urllib.request.urlretrieve(request.image_url, str(tmp))
            return str(tmp)
        except Exception as e:
            logger.error(f"Failed to download image from URL: {e}")
            raise ValueError(f"Failed to download image: {e}") from e

    return None


class VideoService:
    """Synchronous video generation service. All methods run on the MLX worker thread."""

    def generate_video(
        self, request: VideoGenerationRequest, base_url: str = ""
    ) -> VideoObject:
        from mlx_video.models.ltx_2.generate import generate_video

        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        timestamp = int(time.time())
        output_filename = f"video_{timestamp}_{seed}.mp4"
        output_path = str(_VIDEO_OUTPUT_DIR / output_filename)

        image_path = _resolve_image(request)
        is_i2v = image_path is not None
        mode = "I2V" if is_i2v else "T2V"

        logger.info(
            f"[{mode}] Generating video: model={request.model}, "
            f"{request.width}x{request.height}, frames={request.num_frames}, "
            f"pipeline={request.pipeline.value}, seed={seed}"
        )

        extra = request.get_extra_params()
        steps = request.num_inference_steps
        if steps is None:
            steps = 30 if request.pipeline != VideoPipeline.DISTILLED else 40

        gen_kwargs = {
            "model_repo": request.model,
            "text_encoder_repo": extra.get("text_encoder_repo"),
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
            "tiling": request.tiling,
            "stream": False,
        }

        if request.negative_prompt is not None:
            gen_kwargs["negative_prompt"] = request.negative_prompt

        if is_i2v:
            gen_kwargs["image"] = image_path
            gen_kwargs["image_strength"] = request.image_strength

        for key in ("lora_path", "lora_strength", "enhance_prompt", "spatial_upscaler"):
            if key in extra:
                gen_kwargs[key] = extra[key]

        t0 = time.perf_counter()
        generate_video(**gen_kwargs)
        elapsed = time.perf_counter() - t0

        logger.info(f"[{mode}] Video generated in {elapsed:.1f}s: {output_path}")

        # Clean up temp I2V input image
        if image_path:
            Path(image_path).unlink(missing_ok=True)

        if request.response_format == VideoResponseFormat.B64_JSON:
            video_bytes = Path(output_path).read_bytes()
            b64 = base64.b64encode(video_bytes).decode()
            Path(output_path).unlink(missing_ok=True)
            logger.info(f"[{mode}] Encoded video: {len(b64)} chars base64")
            return VideoObject(b64_json=b64, revised_prompt=request.prompt)

        url = f"{base_url}/v1/videos/files/{output_filename}" if base_url else f"file://{output_path}"
        return VideoObject(url=url, revised_prompt=request.prompt)

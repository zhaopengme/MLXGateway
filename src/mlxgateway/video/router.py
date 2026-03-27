import asyncio
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from ..models.error import ErrorDetail, ErrorResponse
from ..utils.gpu import gpu_inference, run_on_mlx_thread
from ..utils.logger import logger
from ..utils.static import TEMP_DIR, get_base_url
from .schema import VideoGenerationRequest, VideoGenerationResponse, VideoPipeline, VideoTiling
from .service import VideoService, resolve_media

router = APIRouter(prefix="/v1", tags=["videos"])

_service = VideoService()


async def _save_upload(upload: UploadFile, label: str) -> str:
    """Save an uploaded file to TEMP_DIR. Returns the temp file path."""
    suffix = Path(upload.filename or "").suffix or ".bin"
    tmp = TEMP_DIR / f"{label}_{uuid.uuid4().hex}{suffix}"
    content = await upload.read()
    tmp.write_bytes(content)
    return str(tmp)


@router.post("/videos/generations")
async def create_video(
    request: VideoGenerationRequest, http_request: Request
) -> VideoGenerationResponse:
    first_image_path = None
    last_image_path = None
    audio_file_path = None
    try:
        has_any_image = any([request.image, request.image_url, request.end_image, request.end_image_url])
        has_audio_input = bool(request.audio_file or request.audio_file_url)
        parts = []
        if has_any_image:
            parts.append("I2V")
        if has_audio_input:
            parts.append("A2V")
        mode = "+".join(parts) if parts else "T2V"

        logger.info(
            f"Video generation request [{mode}]: model={request.model}, "
            f"{request.width}x{request.height}, frames={request.num_frames}"
        )
        base_url = get_base_url(http_request)

        if has_any_image or has_audio_input:
            first_image_path, last_image_path, audio_file_path = await asyncio.to_thread(
                resolve_media, request
            )

        async with gpu_inference("video"):
            video_obj = await run_on_mlx_thread(
                _service.generate_video, request, base_url,
                first_image_path, last_image_path, audio_file_path,
            )

        return VideoGenerationResponse(
            created=int(time.time()),
            data=[video_obj],
        )

    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error=ErrorDetail(
                    message="Server is busy. Request timed out waiting for GPU resources.",
                    type="server_error",
                    code="timeout",
                )
            ).model_dump(),
        )
    except ValueError as ve:
        logger.error(f"Video validation error: {ve}")
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    message=str(ve),
                    type="invalid_request_error",
                    code="invalid_value",
                )
            ).model_dump(),
        )
    except Exception as e:
        logger.error(f"Video generation error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    message="An unexpected error occurred while generating video.",
                    type="server_error",
                    code="internal_error",
                )
            ).model_dump(),
        )
    finally:
        if first_image_path:
            Path(first_image_path).unlink(missing_ok=True)
        if last_image_path:
            Path(last_image_path).unlink(missing_ok=True)
        if audio_file_path:
            Path(audio_file_path).unlink(missing_ok=True)


@router.post("/videos/generations/upload")
async def create_video_upload(
    http_request: Request,
    prompt: str = Form(...),
    model: str = Form(default="prince-canuma/LTX-2.3-distilled"),
    text_encoder_repo: Optional[str] = Form(default="mlx-community/gemma-3-12b-it-bf16"),
    width: int = Form(default=512),
    height: int = Form(default=512),
    num_frames: int = Form(default=97),
    fps: int = Form(default=24),
    seed: Optional[int] = Form(default=None),
    pipeline: str = Form(default="distilled"),
    negative_prompt: Optional[str] = Form(default=None),
    num_inference_steps: Optional[int] = Form(default=None),
    cfg_scale: float = Form(default=3.0),
    response_format: str = Form(default="url"),
    image_strength: float = Form(default=1.0),
    image_frame_idx: int = Form(default=0),
    audio_start_time: float = Form(default=0.0),
    tiling: str = Form(default="auto"),
    image: Optional[UploadFile] = File(default=None),
    end_image: Optional[UploadFile] = File(default=None),
    audio_file: Optional[UploadFile] = File(default=None),
) -> VideoGenerationResponse:
    """Generate video with file uploads (multipart/form-data).

    Accepts image and audio files directly instead of base64/URL.
    Same generation capabilities as the JSON endpoint.
    """
    first_image_path = None
    last_image_path = None
    audio_file_path = None
    try:
        # Save uploaded files
        if image:
            first_image_path = await _save_upload(image, "i2v_first")
        if end_image:
            last_image_path = await _save_upload(end_image, "i2v_last")
        if audio_file:
            audio_file_path = await _save_upload(audio_file, "a2v_audio")

        is_i2v = first_image_path is not None or last_image_path is not None
        is_a2v = audio_file_path is not None
        parts = []
        if is_i2v:
            parts.append("I2V")
        if is_a2v:
            parts.append("A2V")
        mode = "+".join(parts) if parts else "T2V"

        logger.info(
            f"Video upload request [{mode}]: model={model}, "
            f"{width}x{height}, frames={num_frames}"
        )

        # Validate dual-frame constraint (schema can't see uploaded files)
        if first_image_path and last_image_path and num_frames < 9:
            raise ValueError("Dual-frame conditioning (image + end_image) requires at least 9 frames")

        # Build a VideoGenerationRequest for validation and service
        try:
            request_obj = VideoGenerationRequest(
                prompt=prompt,
                model=model,
                text_encoder_repo=text_encoder_repo,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=fps,
                seed=seed,
                pipeline=VideoPipeline(pipeline),
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                cfg_scale=cfg_scale,
                response_format=response_format,
                image_strength=image_strength,
                image_frame_idx=image_frame_idx,
                audio=not is_a2v,
                audio_start_time=audio_start_time,
                tiling=VideoTiling(tiling),
            )
        except (ValidationError, ValueError) as e:
            raise ValueError(str(e)) from e

        base_url = get_base_url(http_request)

        async with gpu_inference("video"):
            video_obj = await run_on_mlx_thread(
                _service.generate_video, request_obj, base_url,
                first_image_path, last_image_path, audio_file_path,
            )

        return VideoGenerationResponse(
            created=int(time.time()),
            data=[video_obj],
        )

    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error=ErrorDetail(
                    message="Server is busy. Request timed out waiting for GPU resources.",
                    type="server_error",
                    code="timeout",
                )
            ).model_dump(),
        )
    except ValueError as ve:
        logger.error(f"Video upload validation error: {ve}")
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    message=str(ve),
                    type="invalid_request_error",
                    code="invalid_value",
                )
            ).model_dump(),
        )
    except Exception as e:
        logger.error(f"Video upload generation error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    message="An unexpected error occurred while generating video.",
                    type="server_error",
                    code="internal_error",
                )
            ).model_dump(),
        )
    finally:
        if first_image_path:
            Path(first_image_path).unlink(missing_ok=True)
        if last_image_path:
            Path(last_image_path).unlink(missing_ok=True)
        if audio_file_path:
            Path(audio_file_path).unlink(missing_ok=True)

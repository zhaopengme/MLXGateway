import asyncio
import time
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..models.error import ErrorDetail, ErrorResponse
from ..utils.gpu import gpu_inference, run_on_mlx_thread
from ..utils.logger import logger
from .schema import VideoGenerationRequest, VideoGenerationResponse
from .service import VideoService, resolve_media

router = APIRouter(prefix="/v1", tags=["videos"])

_service = VideoService()


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
        base_url = str(http_request.base_url).rstrip("/")

        # Resolve all media files before acquiring GPU semaphore
        if has_any_image or has_audio_input:
            first_image_path, last_image_path, audio_file_path = await asyncio.to_thread(
                resolve_media, request
            )

        async with gpu_inference("video"):
            video_obj = await run_on_mlx_thread(
                _service.generate_video, request, base_url,
                first_image_path, last_image_path, audio_file_path,
            )
            first_image_path = last_image_path = audio_file_path = None

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

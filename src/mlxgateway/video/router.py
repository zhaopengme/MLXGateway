import asyncio
import time
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

from ..models.error import ErrorDetail, ErrorResponse
from ..utils.gpu import gpu_inference, run_on_mlx_thread
from ..utils.logger import logger
from .schema import VideoGenerationRequest, VideoGenerationResponse
from .service import VideoService, _VIDEO_OUTPUT_DIR

router = APIRouter(prefix="/v1", tags=["videos"])

_service = VideoService()


@router.get("/videos/files/{filename}")
async def serve_generated_video(filename: str):
    """Serve a previously generated video file."""
    file_path = (_VIDEO_OUTPUT_DIR / filename).resolve()
    if not file_path.is_relative_to(_VIDEO_OUTPUT_DIR.resolve()):
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    message="Invalid filename",
                    type="invalid_request_error",
                    code="invalid_value",
                )
            ).model_dump(),
        )
    if not file_path.exists() or not file_path.is_file():
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error=ErrorDetail(
                    message=f"Video file '{filename}' not found",
                    type="invalid_request_error",
                    code="file_not_found",
                )
            ).model_dump(),
        )
    ext = file_path.suffix.lower()
    media_types = {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime"}
    media_type = media_types.get(ext, "application/octet-stream")
    return FileResponse(file_path, media_type=media_type)


@router.post("/videos/generations")
async def create_video(request: VideoGenerationRequest, http_request: Request):
    try:
        mode = "I2V" if (request.image or request.image_url) else "T2V"
        logger.info(
            f"Video generation request [{mode}]: model={request.model}, "
            f"{request.width}x{request.height}, frames={request.num_frames}"
        )
        base_url = str(http_request.base_url).rstrip("/")

        async with gpu_inference("video"):
            video_obj = await run_on_mlx_thread(
                _service.generate_video, request, base_url
            )

        return VideoGenerationResponse(
            created=int(time.time()),
            data=[video_obj],
        ).model_dump()

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

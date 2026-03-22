import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from starlette.responses import PlainTextResponse

from ...models.error import ErrorDetail, ErrorResponse
from ...utils.gpu import gpu_inference, run_on_mlx_thread
from ...utils.logger import logger
from .schema import ResponseFormat, STTRequestForm, TranscriptionResponse
from .service import STTService

router = APIRouter(prefix="/v1", tags=["speech-to-text"])

_stt_service = STTService()


@router.post("/audio/transcriptions", response_model=TranscriptionResponse)
async def create_transcription(request: STTRequestForm = Depends()):
    audio_path = None
    try:
        # Save uploaded file before acquiring GPU semaphore so we don't hold
        # the semaphore during I/O. The path is tracked for cleanup on error.
        audio_path = await _stt_service.save_upload_file(request.file)
        async with gpu_inference("audio"):
            result = await run_on_mlx_thread(_stt_service.transcribe_sync, request, audio_path)
            audio_path = None  # transcribe_sync owns cleanup on success
        if request.response_format == ResponseFormat.TEXT:
            return PlainTextResponse(content=result)
        return JSONResponse(content=result)
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
    except Exception as e:
        logger.exception("STT transcription failed: %s", e)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    message="An unexpected error occurred while transcribing audio.",
                    type="server_error",
                    code="internal_error",
                )
            ).model_dump(),
        )
    finally:
        if audio_path:
            Path(audio_path).unlink(missing_ok=True)

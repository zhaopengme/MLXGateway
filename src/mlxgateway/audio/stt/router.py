import asyncio

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from starlette.responses import PlainTextResponse

from ...models.error import ErrorDetail, ErrorResponse
from ...utils.gpu import gpu_inference
from ...utils.logger import logger
from .schema import ResponseFormat, STTRequestForm, TranscriptionResponse
from .service import STTService

router = APIRouter(prefix="/v1", tags=["speech-to-text"])

_stt_service = STTService()


@router.post("/audio/transcriptions", response_model=TranscriptionResponse)
async def create_transcription(request: STTRequestForm = Depends()):
    try:
        async with gpu_inference():
            result = await _stt_service.transcribe(request)
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

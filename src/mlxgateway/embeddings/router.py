import asyncio
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..models.error import ErrorDetail, ErrorResponse
from ..utils.gpu import gpu_inference
from ..utils.logger import logger
from .schema import EmbeddingData, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage
from .service import generate_embeddings

router = APIRouter(prefix="/v1", tags=["embeddings"])


def _error(status_code: int, message: str, code: str, param: str = None):
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error=ErrorDetail(message=message, type="invalid_request_error", code=code, param=param)
        ).model_dump(),
    )


@router.post("/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    try:
        if request.encoding_format and request.encoding_format != "float":
            return _error(400, f"encoding_format '{request.encoding_format}' is not supported. Use 'float'.", "invalid_value", "encoding_format")

        texts = [request.input] if isinstance(request.input, str) else request.input
        texts = [t for t in texts if t is not None and str(t).strip()]
        if not texts:
            return _error(400, "Input must not be empty.", "invalid_value", "input")

        t0 = time.perf_counter()
        async with gpu_inference():
            embeddings, total_tokens = await asyncio.to_thread(
                generate_embeddings, request.model, texts
            )
        elapsed = time.perf_counter() - t0

        dim = len(embeddings[0]) if embeddings else 0
        logger.info(
            f"[{request.model}] embeddings={len(texts)} tokens={total_tokens} "
            f"dim={dim} time={elapsed:.2f}s"
        )

        return EmbeddingResponse(
            data=[
                EmbeddingData(embedding=emb, index=i)
                for i, emb in enumerate(embeddings)
            ],
            model=request.model,
            usage=EmbeddingUsage(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens,
            ),
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
    except Exception as e:
        logger.error(f"Embedding error: {e}", exc_info=True)
        error_msg = str(e)
        if "not found" in error_msg.lower() or "no such" in error_msg.lower():
            return _error(400, error_msg, "model_not_found", "model")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(message=error_msg, type="server_error", code="internal_error")
            ).model_dump(),
        )

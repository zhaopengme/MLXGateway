import argparse
import asyncio
import os

from contextlib import asynccontextmanager

import setproctitle
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .audio.stt.router import router as stt_router
from .audio.tts.router import router as tts_router
from .chat.router import router as chat_router
from .config import Config, set_config
from .embeddings.router import router as embeddings_router
from .images.router import router as images_router
from .middleware.auth import APIKeyAuthMiddleware
from .middleware.logging import RequestResponseLoggingMiddleware
from .models.router import router as models_router
from .utils.gpu import get_gpu_semaphore
from .utils.logger import logger, set_logger_level

def _save_all_caches(force: bool = False) -> None:
    """Save prompt caches for all loaded LLM models. Runs on the MLX worker thread."""
    from .models.cache import get_model_cache
    from .utils.gpu import get_mlx_executor  # noqa: F401 (imported for context)
    cache = get_model_cache()
    for model in cache.get_llm_models_for_cache_save():
        try:
            model.save_cache(force=force)
        except Exception as e:
            logger.warning(f"Cache save failed for {model.model_id}: {e}")


async def _periodic_cache_saver(interval: int = 300):
    """Background task: save all loaded model prompt caches every `interval` seconds."""
    from .utils.gpu import run_on_mlx_thread
    while True:
        await asyncio.sleep(interval)
        try:
            await run_on_mlx_thread(_save_all_caches)
        except Exception as e:
            logger.warning(f"Periodic cache saver error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize GPU semaphore on startup to avoid edge cases
    get_gpu_semaphore()
    # Start background task for periodic prompt cache saves
    saver_task = asyncio.create_task(_periodic_cache_saver(interval=300))
    yield
    # Cancel background saver
    saver_task.cancel()
    try:
        await saver_task
    except asyncio.CancelledError:
        pass
    # Graceful shutdown: force-save all prompt caches on the MLX thread
    logger.info("Shutting down MLX Gateway...")
    try:
        from .utils.gpu import run_on_mlx_thread
        import functools
        await run_on_mlx_thread(functools.partial(_save_all_caches, force=True))
        logger.info("Prompt caches saved.")
    except Exception as e:
        logger.warning(f"Shutdown cleanup error: {e}")

app = FastAPI(title="MLX Gateway", lifespan=lifespan)


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error on {request.method} {request.url.path}")
    logger.error(f"Request body: {await request.body()}")
    logger.error(f"Validation errors: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body},
    )

app.include_router(chat_router)
app.include_router(models_router)
app.include_router(images_router)
app.include_router(tts_router)
app.include_router(stt_router)
app.include_router(embeddings_router)


def build_parser():
    parser = argparse.ArgumentParser(description="MLX Gateway")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8008,
        help="Port to bind the server to (default: 8008)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="debug",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set the logging level (default: info)",
    )
    parser.add_argument(
        "--model-cache-ttl",
        type=int,
        default=600,
        help="Model cache TTL in seconds (default: 600)",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=4,
        help="Maximum number of models to cache (default: 4)",
    )
    parser.add_argument(
        "--model-list-cache",
        type=int,
        default=600,
        help="Model list cache TTL in seconds (default: 600)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (env: API_KEY). If not set, no auth required.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Maximum concurrent GPU inference requests (default: 1)",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=300,
        help="Timeout in seconds for requests waiting to acquire GPU (default: 300)",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Path to reference audio file for TTS voice cloning (env: REF_AUDIO_PATH)",
    )
    return parser


def start():
    setproctitle.setproctitle("MLXGateway")
    
    parser = build_parser()
    args = parser.parse_args()

    config = Config.from_env(
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        model_cache_ttl=args.model_cache_ttl,
        max_models=args.max_models,
        model_list_cache_ttl=args.model_list_cache,
        api_key=args.api_key,
        max_concurrent=args.max_concurrent,
        request_timeout=args.request_timeout,
        ref_audio_path=args.ref_audio,
    )
    set_config(config)

    # Middleware order: CORS (outermost) -> Logging -> Auth (innermost)
    # Starlette executes middleware in reverse addition order (last-added runs first),
    # so we add in this order: CORS, then Logging, then Auth.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestResponseLoggingMiddleware)
    if config.api_key:
        app.add_middleware(APIKeyAuthMiddleware, api_key=config.api_key)

    set_logger_level(logger, config.log_level)

    logger.info(f"Starting MLX Gateway on {config.host}:{config.port}")
    logger.info(f"Model cache: max_size={config.max_models}, ttl={config.model_cache_ttl}s")
    logger.info(f"Model list cache: ttl={config.model_list_cache_ttl}s")
    logger.info(f"API key auth: {'enabled' if config.api_key else 'disabled'}")
    logger.info(f"GPU concurrency: max_concurrent={config.max_concurrent}, request_timeout={config.request_timeout}s")
    if config.ref_audio_path:
        logger.info(f"TTS reference audio: {config.ref_audio_path}")

    uvicorn.run(
        "mlxgateway.main:app",
        host=config.host,
        port=config.port,
        log_level=config.log_level,
        use_colors=True,
    )


if __name__ == "__main__":
    start()


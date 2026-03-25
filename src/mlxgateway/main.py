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

from .config import Config, set_config
from .middleware.auth import APIKeyAuthMiddleware
from .middleware.logging import RequestResponseLoggingMiddleware
from .utils.static import STATIC_DIR, ensure_dirs
from .utils.gpu import get_gpu_semaphore
from .utils.logger import logger, set_logger_level

ROUTER_MAP = {
    "chat": "mlxgateway.chat.router:router",
    "models": "mlxgateway.models.router:router",
    "embedding": "mlxgateway.embeddings.router:router",
    "stt": "mlxgateway.audio.stt.router:router",
    "tts": "mlxgateway.audio.tts.router:router",
    "image": "mlxgateway.images.router:router",
    "video": "mlxgateway.video.router:router",
}


def _import_router(dotted_path: str):
    """Import a router from a dotted module:attribute path."""
    module_path, attr = dotted_path.rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _save_all_caches(force: bool = False) -> None:
    """Save prompt caches for all loaded LLM models. Runs on the MLX worker thread."""
    from .models.cache import get_model_cache
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
    get_gpu_semaphore()
    saver_task = asyncio.create_task(_periodic_cache_saver(interval=300))
    yield
    saver_task.cancel()
    try:
        await saver_task
    except asyncio.CancelledError:
        pass
    logger.info("Shutting down MLX Gateway...")
    try:
        from .embeddings.batcher import shutdown_all_batchers
        await shutdown_all_batchers()
    except Exception as e:
        logger.warning(f"Batcher shutdown error: {e}")
    try:
        from .utils.gpu import run_on_mlx_thread
        import functools
        await asyncio.wait_for(
            run_on_mlx_thread(functools.partial(_save_all_caches, force=True)),
            timeout=30,
        )
        logger.info("Prompt caches saved.")
    except asyncio.TimeoutError:
        logger.warning("Shutdown cache save timed out after 30s")
    except Exception as e:
        logger.warning(f"Shutdown cleanup error: {e}")


def create_app(enabled_routers: list[str] | None = None) -> FastAPI:
    """Create a FastAPI app with selected routers.

    Args:
        enabled_routers: List of router names to register (e.g. ["chat", "embedding"]).
                         None or empty means register all routers.
    """
    application = FastAPI(title="MLX Gateway", lifespan=lifespan)

    @application.get("/health", tags=["health"])
    async def health():
        return {"status": "ok"}

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.error(f"Validation error on {request.method} {request.url.path}")
        logger.error(f"Validation errors: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors(), "body": exc.body},
        )

    names = enabled_routers if enabled_routers else list(ROUTER_MAP.keys())
    for name in names:
        name = name.strip()
        if name in ROUTER_MAP:
            application.include_router(_import_router(ROUTER_MAP[name]))
            logger.info(f"Registered router: {name}")

    ensure_dirs()
    from starlette.staticfiles import StaticFiles
    application.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return application


# Lazy app instance -- only created when accessed (avoids loading all routers
# in worker subprocesses that only need a subset).
app: FastAPI | None = None


def get_app() -> FastAPI:
    global app
    if app is None:
        app = create_app()
    return app


def build_parser():
    parser = argparse.ArgumentParser(description="MLX Gateway")
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=8008,
        help="Port to bind the server to (default: 8008)",
    )
    parser.add_argument(
        "--log-level", type=str, default="debug",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set the logging level (default: info)",
    )
    parser.add_argument(
        "--model-cache-ttl", type=int, default=600,
        help="Model cache TTL in seconds (default: 600)",
    )
    parser.add_argument(
        "--max-models", type=int, default=4,
        help="Maximum number of models to cache (default: 4)",
    )
    parser.add_argument(
        "--model-list-cache", type=int, default=600,
        help="Model list cache TTL in seconds (default: 600)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API key for authentication (env: API_KEY). If not set, no auth required.",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=1,
        help="Maximum concurrent GPU inference requests (default: 1)",
    )
    parser.add_argument(
        "--request-timeout", type=int, default=300,
        help="Timeout in seconds for requests waiting to acquire GPU (default: 300)",
    )
    parser.add_argument(
        "--ref-audio", type=str, default=None,
        help="Path to reference audio file for TTS voice cloning (env: REF_AUDIO_PATH)",
    )
    parser.add_argument(
        "--routers", type=str, default="all",
        help="Comma-separated list of routers to enable: "
             "chat,embedding,stt,tts,image,video,models (default: all)",
    )
    parser.add_argument(
        "--mode", type=str, default="single", choices=["single", "multi"],
        help="Run mode: 'single' (one process) or 'multi' (6 worker processes behind proxy)",
    )
    return parser


def start():
    setproctitle.setproctitle("MLXGateway")

    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "multi":
        from .proxy import start_proxy
        start_proxy(args)
        return

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

    # Create app with selected routers
    global app
    if args.routers != "all":
        app = create_app(args.routers.split(","))
    else:
        app = get_app()

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

    routers_str = args.routers if args.routers != "all" else "all"
    logger.info(f"Starting MLX Gateway on {config.host}:{config.port} [routers={routers_str}]")
    logger.info(f"Model cache: max_size={config.max_models}, ttl={config.model_cache_ttl}s")
    logger.info(f"GPU concurrency: max_concurrent={config.max_concurrent}, request_timeout={config.request_timeout}s")

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level,
        use_colors=True,
    )


if __name__ == "__main__":
    start()

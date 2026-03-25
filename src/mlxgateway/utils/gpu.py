"""GPU inference concurrency control via asyncio.Semaphore.

MLX GPU operations run on dedicated single-thread executors to avoid
thread-safety issues with MLX's Metal GPU context.

The main executor (_mlx_executor) handles LLM, Image, Audio, and Video.
Embedding has its own dedicated executor (_embedding_executor) so that
long-running tasks (e.g., video generation) never block embedding requests.

Each inference type also has its own semaphore for admission control.

Usage:
    async with gpu_inference("llm"):
        result = await run_on_mlx_thread(my_blocking_mlx_fn, arg1, arg2)

    async with gpu_inference("embedding"):
        result = await run_on_embedding_thread(my_embedding_fn, arg1)
"""

import asyncio
import concurrent.futures
import threading
from contextlib import asynccontextmanager
from typing import Callable, Literal, TypeVar

from .logger import logger

InferenceType = Literal["llm", "embedding", "image", "audio", "video"]

# Per-type semaphores; initialized lazily on first use.
_semaphores: dict[InferenceType, asyncio.Semaphore] = {}

# Global lock for mx.clear_cache() -- must be held when clearing the MLX
# Metal cache from any thread. Prevents concurrent clear_cache calls between
# the main executor and the embedding executor.
mlx_cache_lock = threading.Lock()

# Main executor: LLM, Image, Audio, Video share this thread.
_mlx_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="mlx-worker"
)

# Embedding executor: independent thread so embeddings are never blocked
# by long-running video/LLM tasks on the main executor.
_embedding_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="mlx-embed"
)

T = TypeVar("T")


def get_mlx_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Return the main MLX executor (for LLM streaming, etc.)."""
    return _mlx_executor


def get_embedding_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Return the dedicated embedding executor."""
    return _embedding_executor


def _get_semaphore(inference_type: InferenceType) -> asyncio.Semaphore:
    """Return the semaphore for the given inference type, initializing on first call."""
    if inference_type not in _semaphores:
        from ..config import get_config
        _semaphores[inference_type] = asyncio.Semaphore(get_config().max_concurrent)
    return _semaphores[inference_type]


def get_gpu_semaphore() -> asyncio.Semaphore:
    """Return the LLM semaphore (kept for backward compatibility)."""
    return _get_semaphore("llm")


async def run_on_mlx_thread(fn: Callable[..., T], *args, **kwargs) -> T:
    """Run a blocking MLX function on the main MLX worker thread.

    Used for LLM, Image, Audio, and Video inference.

    WARNING: Never call this from code already running on an MLX worker
    thread. The single-worker executor would deadlock.
    """
    assert not threading.current_thread().name.startswith("mlx-"), (
        "run_on_mlx_thread called from an MLX worker thread -- this would deadlock"
    )
    loop = asyncio.get_running_loop()
    if kwargs:
        import functools
        fn = functools.partial(fn, **kwargs)
    return await loop.run_in_executor(_mlx_executor, fn, *args)


async def run_on_embedding_thread(fn: Callable[..., T], *args, **kwargs) -> T:
    """Run a blocking MLX function on the dedicated embedding worker thread.

    This executor is independent from the main MLX executor, so embedding
    requests are never blocked by video/LLM/image/audio tasks.

    WARNING: Never call this from code already running on an MLX worker thread.
    """
    assert not threading.current_thread().name.startswith("mlx-"), (
        "run_on_embedding_thread called from an MLX worker thread -- this would deadlock"
    )
    loop = asyncio.get_running_loop()
    if kwargs:
        import functools
        fn = functools.partial(fn, **kwargs)
    return await loop.run_in_executor(_embedding_executor, fn, *args)


@asynccontextmanager
async def gpu_inference(inference_type: InferenceType = "llm"):
    """Acquire the semaphore for the given inference type with a configurable timeout.

    Raises asyncio.TimeoutError if the wait exceeds request_timeout seconds.
    Callers should catch this and return HTTP 503.
    """
    from ..config import get_config
    timeout = get_config().request_timeout
    sem = _get_semaphore(inference_type)

    try:
        await asyncio.wait_for(sem.acquire(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"GPU inference [{inference_type}] request timed out waiting for semaphore")
        raise

    try:
        yield
    finally:
        sem.release()

"""GPU inference concurrency control via asyncio.Semaphore.

All MLX GPU operations must run on a single dedicated thread (_mlx_executor)
to avoid thread-safety issues with MLX's Metal GPU context. Using
asyncio.to_thread() or the default executor can cause deadlocks because
MLX is not safe to call from arbitrary threads.

Each inference type (LLM, Embedding, Image, Audio) has its own semaphore so
different request types can execute concurrently without blocking each other.
For example, a long-running LLM chat generation will not block an embedding
request from being processed.

Usage:
    async with gpu_inference("llm"):
        result = await run_on_mlx_thread(my_blocking_mlx_fn, arg1, arg2)
"""

import asyncio
import concurrent.futures
from contextlib import asynccontextmanager
from typing import Callable, Literal, TypeVar

from .logger import logger

InferenceType = Literal["llm", "embedding", "image", "audio"]


def get_mlx_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Return the shared single-thread MLX executor."""
    return _mlx_executor

# Per-type semaphores; initialized lazily on first use.
_semaphores: dict[InferenceType, asyncio.Semaphore] = {}

# Single-threaded executor: all MLX GPU calls are serialized on this thread.
_mlx_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="mlx-worker"
)

T = TypeVar("T")


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
    """Run a blocking MLX function on the dedicated MLX worker thread.

    This avoids blocking the asyncio event loop while ensuring all MLX
    operations execute on the same thread (required for Metal GPU safety).
    Keyword arguments are supported via functools.partial internally.
    """
    loop = asyncio.get_running_loop()
    if kwargs:
        import functools
        fn = functools.partial(fn, **kwargs)
    return await loop.run_in_executor(_mlx_executor, fn, *args)


@asynccontextmanager
async def gpu_inference(inference_type: InferenceType = "llm"):
    """Acquire the semaphore for the given inference type with a configurable timeout.

    Each inference type has an independent semaphore, allowing LLM, embedding,
    image, and audio requests to run concurrently when resources permit.

    Raises asyncio.TimeoutError if the wait exceeds request_timeout seconds.
    Callers should catch this and return HTTP 503.

    Args:
        inference_type: One of "llm", "embedding", "image", "audio".
                        Defaults to "llm" for backward compatibility.
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

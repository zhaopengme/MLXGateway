"""GPU inference concurrency control via asyncio.Semaphore."""

import asyncio
from contextlib import asynccontextmanager

from .logger import logger

_semaphore: asyncio.Semaphore | None = None


def get_gpu_semaphore() -> asyncio.Semaphore:
    """Return the global GPU semaphore, initializing it on first call."""
    global _semaphore
    if _semaphore is None:
        from ..config import get_config
        _semaphore = asyncio.Semaphore(get_config().max_concurrent)
    return _semaphore


@asynccontextmanager
async def gpu_inference():
    """Acquire the GPU semaphore with a configurable timeout.

    Raises asyncio.TimeoutError if the wait exceeds request_timeout seconds.
    Callers should catch this and return HTTP 503.
    """
    from ..config import get_config
    timeout = get_config().request_timeout
    sem = get_gpu_semaphore()

    try:
        await asyncio.wait_for(sem.acquire(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("GPU inference request timed out waiting for semaphore")
        raise

    try:
        yield
    finally:
        sem.release()

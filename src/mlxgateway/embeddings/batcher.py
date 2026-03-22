"""Dynamic batching for embedding requests.

When multiple embedding requests arrive within a short window, they are coalesced
into a single model call, improving throughput significantly compared to
processing each request individually.

Architecture:
    - Each model_id has its own EmbeddingBatcher instance.
    - When a request arrives, it is added to a pending queue.
    - A flush happens when:
        (a) the batch reaches MAX_BATCH_SIZE texts, or
        (b) BATCH_WINDOW_MS milliseconds have elapsed since the first request
            in the current batch.
    - All requests in a batch share a single gpu_inference semaphore acquisition,
      so the semaphore is held for a shorter total time overall.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..utils.logger import logger

# Maximum number of texts to process in one batch.
MAX_BATCH_SIZE = 32

# How long to wait (ms) before flushing an incomplete batch.
BATCH_WINDOW_MS = 20


@dataclass
class _PendingItem:
    texts: List[str]
    future: asyncio.Future


class EmbeddingBatcher:
    """Coalesces concurrent embedding requests into batches for a single model."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._queue: List[_PendingItem] = []
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._total_texts = 0
        # Strong references to in-flight batch tasks to prevent GC before completion.
        self._active_tasks: set = set()

    async def embed(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        """Submit texts for embedding and await the result."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        async with self._lock:
            item = _PendingItem(texts=texts, future=future)
            self._queue.append(item)
            self._total_texts += len(texts)

            if self._total_texts >= MAX_BATCH_SIZE:
                # Batch is full; flush immediately.
                if self._flush_task and not self._flush_task.done():
                    self._flush_task.cancel()
                await self._flush()
            elif self._flush_task is None or self._flush_task.done():
                # Schedule a flush after the window.
                self._flush_task = asyncio.create_task(self._flush_after_window())

        return await future

    async def _flush_after_window(self):
        await asyncio.sleep(BATCH_WINDOW_MS / 1000)
        async with self._lock:
            if self._queue:
                await self._flush()

    async def _flush(self):
        """Process all queued items as one batch. Must be called under self._lock."""
        if not self._queue:
            return

        batch_queue = self._queue[:]
        self._queue = []
        self._total_texts = 0
        self._flush_task = None

        all_texts: List[str] = []
        for item in batch_queue:
            all_texts.extend(item.texts)

        logger.debug(
            f"[Batcher:{self.model_id}] Flushing batch: "
            f"{len(batch_queue)} requests, {len(all_texts)} texts"
        )

        # Run inference outside the lock so new requests can queue up.
        # Keep a strong reference so the task is not GC'd before completion.
        task = asyncio.get_running_loop().create_task(
            self._run_batch(batch_queue, all_texts)
        )
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)

    async def _run_batch(self, batch_queue: List[_PendingItem], all_texts: List[str]):
        from ..utils.gpu import gpu_inference, run_on_mlx_thread
        from .service import generate_embeddings

        try:
            async with gpu_inference("embedding"):
                all_embeddings, total_tokens = await run_on_mlx_thread(
                    generate_embeddings, self.model_id, all_texts
                )

            # Distribute results back to each waiting request.
            idx = 0
            for item in batch_queue:
                n = len(item.texts)
                embeddings_slice = all_embeddings[idx: idx + n]
                # Estimate token share proportionally.
                tokens_share = int(total_tokens * n / len(all_texts)) if all_texts else 0
                if not item.future.done():
                    item.future.set_result((embeddings_slice, tokens_share))
                idx += n

        except Exception as exc:
            for item in batch_queue:
                if not item.future.done():
                    item.future.set_exception(exc)


# Global registry: one batcher per model_id.
_batchers: Dict[str, EmbeddingBatcher] = {}
_batchers_lock: Optional[asyncio.Lock] = None


def _get_batchers_lock() -> asyncio.Lock:
    global _batchers_lock
    if _batchers_lock is None:
        _batchers_lock = asyncio.Lock()
    return _batchers_lock


async def get_batcher(model_id: str) -> EmbeddingBatcher:
    """Return the batcher for the given model, creating it if necessary."""
    async with _get_batchers_lock():
        if model_id not in _batchers:
            _batchers[model_id] = EmbeddingBatcher(model_id)
        return _batchers[model_id]


async def shutdown_all_batchers() -> None:
    """Cancel all pending embedding futures. Call during server shutdown."""
    async with _get_batchers_lock():
        for batcher in _batchers.values():
            async with batcher._lock:
                for item in batcher._queue:
                    if not item.future.done():
                        item.future.cancel()
                batcher._queue.clear()
                batcher._total_texts = 0
        _batchers.clear()

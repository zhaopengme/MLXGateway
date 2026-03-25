import gc
import threading
from collections import OrderedDict
from typing import List, Tuple

import mlx.core as mx
from mlx_embeddings.utils import load

from ..utils.logger import logger

_cache: OrderedDict[str, Tuple] = OrderedDict()
_lock = threading.Lock()
_MAX_CACHE = 2


def _get_model(model_id: str):
    with _lock:
        if model_id in _cache:
            _cache.move_to_end(model_id)
            return _cache[model_id]

    logger.info(f"Loading embedding model: {model_id}")
    model, tokenizer = load(model_id)
    logger.info(f"Embedding model loaded: {model_id}")

    with _lock:
        if model_id in _cache:
            _cache.move_to_end(model_id)
            return _cache[model_id]
        if len(_cache) >= _MAX_CACHE:
            evict_key, _ = _cache.popitem(last=False)
            mx.clear_cache()
            gc.collect()
            logger.info(f"Evicted embedding model: {evict_key}")
        _cache[model_id] = (model, tokenizer)
    return model, tokenizer


def generate_embeddings(
    model_id: str,
    texts: List[str],
) -> Tuple[List[List[float]], int]:
    """Generate embeddings for a list of texts.
    
    Returns (embeddings_list, total_tokens).
    """
    model, tokenizer = _get_model(model_id)

    inputs = tokenizer._tokenizer(
        texts, return_tensors="mlx", padding=True, truncation=True, max_length=512
    )
    total_tokens = inputs["input_ids"].size
    outputs = model(
        inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
    )

    embeddings = outputs.text_embeds
    mx.eval(embeddings)

    return embeddings.tolist(), int(total_tokens)

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

from ..utils.logger import logger
from .loader import MLXModel

if TYPE_CHECKING:
    from ..chat.generator import ChatGenerator
    from ..vlm.loader import VLMModel
    from ..vlm.generator import VLMGenerator


@dataclass(frozen=True)
class CacheKey:
    model_id: str
    adapter_path: Optional[str] = None


class ModelCache:
    def __init__(self, max_size: Optional[int] = None, ttl_seconds: Optional[int] = None):
        from ..config import get_config
        config = get_config()
        
        self._cache: Dict[CacheKey, MLXModel] = {}
        self._vlm_cache: Dict[CacheKey, "VLMModel"] = {}
        self._generators: Dict[CacheKey, "ChatGenerator"] = {}
        self._vlm_generators: Dict[CacheKey, "VLMGenerator"] = {}
        self._access_times: Dict[CacheKey, float] = {}
        self._lock = threading.Lock()
        self._key_locks: Dict[CacheKey, threading.RLock] = {}
        self.max_size = max_size if max_size is not None else config.max_models
        self.ttl = ttl_seconds if ttl_seconds is not None else config.model_cache_ttl
        self.config = config

        if self.ttl > 0:
            threading.Thread(target=self._cleanup_loop, daemon=True).start()

    def _get_key_lock(self, key: CacheKey) -> threading.RLock:
        """Get or create a per-key reentrant lock to prevent duplicate loading."""
        if key not in self._key_locks:
            self._key_locks[key] = threading.RLock()
        return self._key_locks[key]

    def _evict(self, key: CacheKey, reason: str) -> list:
        """Remove a model from all caches (must be called under self._lock).

        Returns a list of model objects that need .unload() called (which
        involves MLX GPU operations and must run on the MLX worker thread).
        """
        to_unload = []
        if key in self._cache:
            to_unload.append(self._cache.pop(key))
        if key in self._vlm_cache:
            to_unload.append(self._vlm_cache.pop(key))

        self._generators.pop(key, None)
        self._vlm_generators.pop(key, None)
        self._access_times.pop(key, None)
        self._key_locks.pop(key, None)
        logger.info(f"{reason}: {key.model_id}")
        return to_unload

    def get_model(
        self,
        model_id: str,
        adapter_path: Optional[str] = None,
        use_cache: bool = True,
        max_kv_size: Optional[int] = None,
    ) -> MLXModel:
        key = CacheKey(model_id, adapter_path)
        
        with self._lock:
            if key in self._cache:
                self._access_times[key] = time.time()
                return self._cache[key]
            key_lock = self._get_key_lock(key)

        with key_lock:
            # Double-check after acquiring per-key lock
            with self._lock:
                if key in self._cache:
                    self._access_times[key] = time.time()
                    return self._cache[key]
                
                if self.max_size > 0 and len(self._cache) >= self.max_size:
                    candidates = {k: t for k, t in self._access_times.items() if k in self._cache}
                    if candidates:
                        for m in self._evict(min(candidates, key=candidates.get), "Evicted LLM"):
                            m.unload()

            # Determine cache settings
            final_use_cache = use_cache and self.config.enable_cache_by_default
            final_max_kv_size = max_kv_size if max_kv_size is not None else self.config.default_max_kv_size
            
            model = MLXModel.load(
                model_id,
                adapter_path,
                use_cache=final_use_cache,
                max_kv_size=final_max_kv_size,
            )
            
            with self._lock:
                self._cache[key] = model
                self._access_times[key] = time.time()
            
            return model
    
    def get_generator(
        self, 
        model_id: str, 
        adapter_path: Optional[str] = None,
        use_cache: bool = True,
        max_kv_size: Optional[int] = None,
    ) -> "ChatGenerator":
        """
        Get or create a ChatGenerator for the specified model.
        
        Args:
            model_id: Model identifier
            adapter_path: Optional adapter path
            use_cache: Whether to enable prompt cache
            max_kv_size: Maximum KV cache size (passed to make_prompt_cache)
            
        Returns:
            ChatGenerator instance
        """
        key = CacheKey(model_id, adapter_path)
        
        with self._lock:
            if key in self._generators:
                self._access_times[key] = time.time()
                return self._generators[key]
            key_lock = self._get_key_lock(key)
        
        with key_lock:
            with self._lock:
                if key in self._generators:
                    self._access_times[key] = time.time()
                    return self._generators[key]
            
            from ..chat.generator import ChatGenerator
            
            model = self.get_model(model_id, adapter_path, use_cache, max_kv_size)
            generator = ChatGenerator(model, model_id)
            
            with self._lock:
                self._generators[key] = generator
            
            return generator

    def get_vlm_model(self, model_id: str, adapter_path: Optional[str] = None) -> "VLMModel":
        """
        Get or load a VLM model.
        
        Args:
            model_id: Model identifier
            adapter_path: Optional adapter path
            
        Returns:
            VLMModel instance
        """
        key = CacheKey(model_id, adapter_path)
        
        with self._lock:
            if key in self._vlm_cache:
                self._access_times[key] = time.time()
                return self._vlm_cache[key]
            key_lock = self._get_key_lock(key)
        
        with key_lock:
            with self._lock:
                if key in self._vlm_cache:
                    self._access_times[key] = time.time()
                    return self._vlm_cache[key]
                
                if self.max_size > 0 and len(self._vlm_cache) + len(self._cache) >= self.max_size:
                    all_model_keys = set(self._cache) | set(self._vlm_cache)
                    candidates = {k: t for k, t in self._access_times.items() if k in all_model_keys}
                    if candidates:
                        for m in self._evict(min(candidates, key=candidates.get), "Evicted"):
                            m.unload()

            from ..vlm.loader import VLMModel
            model = VLMModel.load(model_id, adapter_path)
            
            with self._lock:
                self._vlm_cache[key] = model
                self._access_times[key] = time.time()
            
            return model
    
    def get_vlm_generator(
        self, 
        model_id: str, 
        adapter_path: Optional[str] = None
    ) -> "VLMGenerator":
        """
        Get or create a VLMGenerator for the specified model.
        
        Args:
            model_id: Model identifier
            adapter_path: Optional adapter path
            
        Returns:
            VLMGenerator instance
        """
        key = CacheKey(model_id, adapter_path)
        
        with self._lock:
            if key in self._vlm_generators:
                self._access_times[key] = time.time()
                return self._vlm_generators[key]
            key_lock = self._get_key_lock(key)
        
        with key_lock:
            with self._lock:
                if key in self._vlm_generators:
                    self._access_times[key] = time.time()
                    return self._vlm_generators[key]
            
            from ..vlm.generator import VLMGenerator
            
            model = self.get_vlm_model(model_id, adapter_path)
            generator = VLMGenerator(model)
            
            with self._lock:
                self._vlm_generators[key] = generator
            
            return generator
    
    def get_loaded_model_ids(self) -> List[str]:
        with self._lock:
            return [key.model_id for key in list(self._cache.keys()) + list(self._vlm_cache.keys())]

    def get_llm_models_for_cache_save(self) -> List[MLXModel]:
        """Return a snapshot of currently loaded LLM models, safe to iterate outside the lock."""
        with self._lock:
            return list(self._cache.values())
    
    def _cleanup_loop(self) -> None:
        from ..utils.gpu import _mlx_executor
        while True:
            time.sleep(60)
            try:
                if self.ttl <= 0:
                    continue
                models_to_unload = []
                with self._lock:
                    now = time.time()
                    for key in [k for k, t in self._access_times.items() if now - t > self.ttl]:
                        if key in self._cache or key in self._vlm_cache:
                            models_to_unload.extend(self._evict(key, "Expired"))
                # Unload on the MLX worker thread (involves MLX GPU operations).
                if models_to_unload:
                    def _unload_all(models=models_to_unload):
                        for m in models:
                            m.unload()
                    _mlx_executor.submit(_unload_all)
            except Exception as e:
                logger.warning(f"Cleanup loop error (will retry): {e}")


_model_cache_instance: Optional[ModelCache] = None


def get_model_cache() -> ModelCache:
    global _model_cache_instance
    if _model_cache_instance is None:
        from ..config import get_config
        config = get_config()
        _model_cache_instance = ModelCache(config.max_models, config.model_cache_ttl)
    return _model_cache_instance

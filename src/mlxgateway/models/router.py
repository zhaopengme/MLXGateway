import importlib
import json
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from huggingface_hub import scan_cache_dir

from ..utils.logger import logger
from .error import ErrorDetail, ErrorResponse
from .schema import Model, ModelList

router = APIRouter(prefix="/v1", tags=["models"])

_cache: Optional[Tuple[ModelList, float]] = None
_lock = threading.Lock()


class ModelScanner(ABC):
    """Base class for model scanners"""
    
    @abstractmethod
    def should_include(self, repo, revision) -> bool:
        """Determine if a repo should be included"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Scanner name for logging"""
        pass


class LLMScanner(ModelScanner):
    """Scans for LLM models with valid MLX architecture"""
    
    @property
    def name(self) -> str:
        return "LLM"
    
    def should_include(self, repo, revision) -> bool:
        config_file = next((f for f in revision.files if f.file_name == "config.json"), None)
        if not config_file:
            return False
        
        try:
            with open(config_file.file_path) as f:
                config = json.load(f)
            
            model_type = config.get("model_type")
            if not model_type:
                return False
            
            arch = importlib.import_module(f"mlx_lm.models.{model_type}")
            return hasattr(arch, "Model") and hasattr(arch, "ModelArgs")
        except Exception:
            return False


class AudioScanner(ModelScanner):
    """Dynamically scans for audio models supported by mlx-audio"""
    
    def __init__(self):
        self._patterns = self._discover_audio_models()
    
    @property
    def name(self) -> str:
        return "Audio"
    
    def _discover_audio_models(self) -> List[str]:
        """Discover supported audio model types from mlx_audio package"""
        import mlx_audio
        mlx_audio_path = os.path.dirname(mlx_audio.__file__)
        
        patterns = []
        for submodule in ["stt", "tts"]:
            models_path = os.path.join(mlx_audio_path, submodule, "models")
            if os.path.exists(models_path):
                patterns.extend([
                    item for item in os.listdir(models_path)
                    if os.path.isdir(os.path.join(models_path, item)) and not item.startswith("__")
                ])
        
        return patterns
    
    def should_include(self, repo, revision) -> bool:
        repo_lower = repo.repo_id.lower()
        return any(pattern in repo_lower for pattern in self._patterns)


class GGUFScanner(ModelScanner):
    """Scans HuggingFace cache for repos that contain .gguf files."""

    @property
    def name(self) -> str:
        return "GGUF"

    def should_include(self, repo, revision) -> bool:
        return any(f.file_name.lower().endswith(".gguf") for f in revision.files)


# Model scanner registry
SCANNERS: List[ModelScanner] = [
    LLMScanner(),
    AudioScanner(),
    GGUFScanner(),
]


def scan_models(scanners: List[ModelScanner]) -> List[Model]:
    """Scan HuggingFace cache for models using provided scanners"""
    from .cache import get_model_cache
    
    models = []
    scanner_counts = {scanner.name: 0 for scanner in scanners}
    loaded_ids = set(get_model_cache().get_loaded_model_ids())
    
    try:
        for repo in scan_cache_dir().repos:
            if repo.repo_type != "model":
                continue
            
            revision = next(iter(repo.revisions), None)
            if not revision:
                continue
            
            for scanner in scanners:
                try:
                    if scanner.should_include(repo, revision):
                        owner = repo.repo_id.split("/")[0] if "/" in repo.repo_id else repo.repo_id
                        models.append(Model(
                            id=repo.repo_id,
                            created=int(repo.last_modified),
                            owned_by=owner,
                            loaded=repo.repo_id in loaded_ids,
                        ))
                        scanner_counts[scanner.name] += 1
                        break  # Model already added, skip other scanners
                except Exception as e:
                    logger.error(f"Error scanning {repo.repo_id} with {scanner.name}: {e}")
        
        # Log results
        summary = ", ".join(f"{count} {name}" for name, count in scanner_counts.items())
        logger.info(f"Found models: {summary}")
    except Exception as e:
        logger.error(f"Model scan error: {e}")
    
    return models


def get_models(ttl: int) -> ModelList:
    global _cache

    with _lock:
        now = time.time()
        if _cache:
            cached_list, cached_at = _cache
            if now - cached_at < ttl:
                logger.debug(f"Model cache hit (age: {now - cached_at:.1f}s)")
                from .cache import get_model_cache
                loaded_ids = set(get_model_cache().get_loaded_model_ids())
                # Return a copy with updated loaded status to avoid mutating the cached objects.
                updated_data = [
                    Model(id=m.id, created=m.created, owned_by=m.owned_by, loaded=m.id in loaded_ids)
                    for m in cached_list.data
                ]
                return ModelList(data=updated_data)

    # Scan outside the lock: reading the HuggingFace cache can be slow and
    # must not block concurrent requests that hit the cache-hit path above.
    logger.info("Scanning HuggingFace cache...")
    models = scan_models(SCANNERS)
    model_list = ModelList(data=models)

    with _lock:
        now = time.time()
        # Another coroutine may have populated the cache while we were scanning;
        # only update if ours is fresher (i.e. the slot is empty or still stale).
        if not _cache or now - _cache[1] >= ttl:
            _cache = (model_list, now)
        else:
            # Use the fresher result from the other coroutine.
            cached_list, _ = _cache
            from .cache import get_model_cache
            loaded_ids = set(get_model_cache().get_loaded_model_ids())
            updated_data = [
                Model(id=m.id, created=m.created, owned_by=m.owned_by, loaded=m.id in loaded_ids)
                for m in cached_list.data
            ]
            return ModelList(data=updated_data)

    return model_list


@router.get("/models", response_model=ModelList)
async def list_models():
    from ..config import get_config
    return get_models(get_config().model_list_cache_ttl)


@router.get("/models/{model_id:path}", response_model=Model)
async def get_model(model_id: str):
    from ..config import get_config
    from .cache import get_model_cache
    
    model_list = get_models(get_config().model_list_cache_ttl)

    for model in model_list.data:
        if model.id == model_id:
            model.loaded = model_id in get_model_cache().get_loaded_model_ids()
            return model

    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            error=ErrorDetail(
                message=f"Model '{model_id}' not found",
                type="invalid_request_error",
                param="model",
                code="model_not_found"
            )
        ).model_dump()
    )

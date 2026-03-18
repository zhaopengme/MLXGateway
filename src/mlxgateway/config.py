import os
from dataclasses import dataclass
from typing import Optional


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


@dataclass
class Config:
    host: str
    port: int
    log_level: str
    model_cache_ttl: int
    max_models: int
    model_list_cache_ttl: int
    api_key: Optional[str]

    # Prompt Cache configuration
    default_max_kv_size: Optional[int]
    enable_cache_by_default: bool

    # GPU concurrency
    max_concurrent: int
    request_timeout: int

    # TTS reference audio
    ref_audio_path: Optional[str]

    @classmethod
    def from_env(
        cls,
        host: Optional[str] = None,
        port: Optional[int] = None,
        log_level: Optional[str] = None,
        model_cache_ttl: Optional[int] = None,
        max_models: Optional[int] = None,
        model_list_cache_ttl: Optional[int] = None,
        api_key: Optional[str] = None,
        default_max_kv_size: Optional[int] = None,
        enable_cache_by_default: Optional[bool] = None,
        max_concurrent: Optional[int] = None,
        request_timeout: Optional[int] = None,
        ref_audio_path: Optional[str] = None,
    ) -> "Config":
        env_enable_cache = os.getenv("ENABLE_CACHE", "true").lower()
        parsed_enable_cache = env_enable_cache in ("true", "1", "yes")

        resolved_max_concurrent = max_concurrent if max_concurrent is not None else _env_int("MAX_CONCURRENT", 1)
        resolved_request_timeout = request_timeout if request_timeout is not None else _env_int("REQUEST_TIMEOUT", 300)

        if resolved_max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        if resolved_request_timeout < 0:
            raise ValueError("request_timeout must be >= 0")

        return cls(
            host=host if host is not None else _env_str("HOST", "127.0.0.1"),
            port=port if port is not None else _env_int("PORT", 8008),
            log_level=log_level if log_level is not None else _env_str("LOG_LEVEL", "info"),
            model_cache_ttl=model_cache_ttl if model_cache_ttl is not None else _env_int("MODEL_CACHE_TTL", 600),
            max_models=max_models if max_models is not None else _env_int("MAX_MODELS", 4),
            model_list_cache_ttl=model_list_cache_ttl if model_list_cache_ttl is not None else _env_int("MODEL_LIST_CACHE_TTL", 600),
            api_key=api_key if api_key is not None else os.getenv("API_KEY"),
            default_max_kv_size=default_max_kv_size if default_max_kv_size is not None else (
                int(os.getenv("DEFAULT_MAX_KV_SIZE")) if os.getenv("DEFAULT_MAX_KV_SIZE") else None
            ),
            enable_cache_by_default=enable_cache_by_default if enable_cache_by_default is not None else parsed_enable_cache,
            max_concurrent=resolved_max_concurrent,
            request_timeout=resolved_request_timeout,
            ref_audio_path=ref_audio_path if ref_audio_path is not None else os.getenv("REF_AUDIO_PATH"),
        )


_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config):
    global _config
    _config = config

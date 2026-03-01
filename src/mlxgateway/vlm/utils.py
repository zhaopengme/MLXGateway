"""Utility functions for VLM content/model detection and error formatting."""

from functools import lru_cache

from mlx_vlm.utils import load_config

from ..utils.logger import logger

# Supported modalities content type mappings
MODEL_MODALITIES = {
    "image": ["image_url", "input_image"],
    "audio": ["input_audio", "audio_url"],
    "video": ["input_video", "video_url"],
}


def detect_multimodal_content(content) -> dict:
    """
    Detect multimodal content types in a message content.
    
    Args:
        content: Message content (string or list of content items)
        
    Returns:
        Dict with detected modalities: {"has_images": bool, "has_audio": bool, "has_video": bool}
    """
    result = {
        "has_images": False,
        "has_audio": False,
        "has_video": False,
        "has_text": False,
    }
    
    if isinstance(content, str):
        result["has_text"] = True
        return result
    
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            
            content_type = item.get("type", "")
            
            if content_type == "text":
                result["has_text"] = True
            elif content_type in MODEL_MODALITIES["image"]:
                result["has_images"] = True
            elif content_type in MODEL_MODALITIES["audio"]:
                result["has_audio"] = True
            elif content_type in MODEL_MODALITIES["video"]:
                result["has_video"] = True
    
    return result


@lru_cache(maxsize=256)
def is_vlm_model(model_id: str) -> bool:
    """
    Determine whether the provided model should be loaded via mlx-vlm.

    The check is heuristic-based and intentionally permissive for common VLM
    config shapes.
    """
    try:
        config = load_config(model_id) or {}
    except Exception as e:
        logger.debug(f"Could not load config to detect VLM model '{model_id}': {e}")
        return False

    vision_markers = (
        "vision_config",
        "vision_tower",
        "image_token_index",
        "mm_projector_type",
        "image_seq_length",
    )
    if any(marker in config for marker in vision_markers):
        return True

    architectures = config.get("architectures") or []
    if not isinstance(architectures, list):
        architectures = [architectures]
    arch_text = " ".join(str(a).lower() for a in architectures)
    return any(token in arch_text for token in ("vision", "vl", "llava", "qwen2vl"))



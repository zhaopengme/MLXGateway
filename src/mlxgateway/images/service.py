import base64
import inspect
import io
import random
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List

from ..utils.logger import logger
from .schema import ImageEditRequest, ImageGenerationRequest, ImageObject, ResponseFormat


def _normalize_model_name(name: str) -> str:
    return name.replace("flux.1", "flux1").replace("flux.2", "flux2").lower()


def _resolve_model_class(model_version: str):
    normalized = _normalize_model_name(model_version)

    if "z-image-turbo" in normalized or "z-turbo" in normalized:
        from mflux.models.z_image import ZImageTurbo
        return ZImageTurbo
    if "z-image" in normalized:
        from mflux.models.z_image import ZImage
        return ZImage
    if "flux2-klein" in normalized:
        from mflux.models.flux2.variants import Flux2Klein
        return Flux2Klein
    if "flux2" in normalized:
        from mflux.models.flux2.variants import Flux2Klein
        return Flux2Klein
    if "flux1-schnell" in normalized or "flux-schnell" in normalized:
        from mflux.models.flux import Flux1Schnell
        return Flux1Schnell
    if "flux1-dev" in normalized or "flux-dev" in normalized:
        from mflux.models.flux import Flux1Dev
        return Flux1Dev

    logger.warning(f"Unknown model: {model_version}, defaulting to Flux2Klein")
    from mflux.models.flux2.variants import Flux2Klein
    return Flux2Klein


def _resolve_edit_class(model_version: str):
    normalized = _normalize_model_name(model_version)

    if "flux2" in normalized and "edit" in normalized:
        from mflux.models.flux2.variants import Flux2KleinEdit
        return Flux2KleinEdit
    if "gpt-image" in normalized:
        from mflux.models.flux2.variants import Flux2KleinEdit
        return Flux2KleinEdit
    if "qwen" in normalized:
        from mflux.models.qwen.variants.edit import QwenImageEdit
        return QwenImageEdit
    if "kontext" in normalized:
        from mflux.models.flux import Flux1Kontext
        return Flux1Kontext

    from mflux.models.flux2.variants import Flux2KleinEdit
    return Flux2KleinEdit


def _build_init_kwargs(model_version: str, params: dict) -> dict:
    init_kwargs = {"quantize": params.get("quantize", 8)}
    normalized = _normalize_model_name(model_version)
    if "flux2-klein-9b" in normalized or ("flux2-klein" in normalized and "9b" in normalized):
        from mflux.models.common.config.model_config import ModelConfig
        init_kwargs["model_config"] = ModelConfig.flux2_klein_9b()
        logger.info("Using FLUX.2-klein-9B model config")
    for key in ["model_path", "lora_paths", "lora_scales"]:
        clean = key.replace("-", "_")
        if key in params:
            init_kwargs[clean] = params[key]
        elif clean in params:
            init_kwargs[clean] = params[clean]
    return init_kwargs


def _add_guidance(generator, gen_kwargs: dict, model_version: str, guidance) -> None:
    if not hasattr(generator, "generate_image"):
        return
    sig = inspect.signature(generator.generate_image)
    if "guidance" not in sig.parameters:
        return
    normalized = _normalize_model_name(model_version)
    gen_kwargs["guidance"] = 1.0 if "flux2" in normalized else guidance


def _pil_to_b64(pil_image, fmt: str = "JPEG", quality: int = 85) -> str:
    buf = io.BytesIO()
    save_kwargs = {"format": fmt}
    if fmt in ("JPEG", "WEBP"):
        save_kwargs["quality"] = quality
    if fmt == "JPEG" and pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")
    pil_image.save(buf, **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Module-level generator / editor cache (persists across requests)
# ---------------------------------------------------------------------------
_gen_cache: Dict[str, object] = {}
_edit_cache: Dict[str, object] = {}
_cache_lock = threading.Lock()
_model_locks: Dict[str, threading.Lock] = {}
_model_locks_lock = threading.Lock()


def _get_model_lock(key: str) -> threading.Lock:
    with _model_locks_lock:
        if key not in _model_locks:
            _model_locks[key] = threading.Lock()
        return _model_locks[key]


def _get_or_create_generator(model_version: str, params: dict):
    with _cache_lock:
        if model_version in _gen_cache:
            logger.debug(f"Image generator cache hit: {model_version}")
            return _gen_cache[model_version]

    with _get_model_lock(f"gen:{model_version}"):
        with _cache_lock:
            if model_version in _gen_cache:
                return _gen_cache[model_version]

        logger.info(f"Loading image model: {model_version}")
        cls = _resolve_model_class(model_version)
        gen = cls(**_build_init_kwargs(model_version, params))
        logger.info(f"Model loaded: {type(gen).__name__}")

        with _cache_lock:
            _gen_cache[model_version] = gen
        return gen


def _get_or_create_editor(model_version: str, params: dict):
    with _cache_lock:
        if model_version in _edit_cache:
            logger.debug(f"Image editor cache hit: {model_version}")
            return _edit_cache[model_version]

    with _get_model_lock(f"edit:{model_version}"):
        with _cache_lock:
            if model_version in _edit_cache:
                return _edit_cache[model_version]

        logger.info(f"Loading edit model: {model_version}")
        cls = _resolve_edit_class(model_version)
        editor = cls(**_build_init_kwargs(model_version, params))
        logger.info(f"Edit model loaded: {type(editor).__name__}")

        with _cache_lock:
            _edit_cache[model_version] = editor
        return editor


# ---------------------------------------------------------------------------
# Service (stateless — cache lives at module level)
# ---------------------------------------------------------------------------
class ImagesService:

    def generate_images(self, request: ImageGenerationRequest) -> List[ImageObject]:
        params = request.get_extra_params()
        generator = _get_or_create_generator(request.model, params)

        width, height = map(int, request.size.split("x"))
        user_seed = params.get("seed")
        steps = params.get("steps", 4)
        guidance = params.get("guidance", 4.0)
        images: List[ImageObject] = []

        for i in range(request.n):
            seed = user_seed if user_seed is not None else random.randint(0, 2**32 - 1)

            logger.info(f"Generating image {i+1}/{request.n}: {width}x{height}, seed={seed}, steps={steps}")

            gen_kwargs = {
                "seed": seed,
                "prompt": request.prompt,
                "num_inference_steps": steps,
                "height": height,
                "width": width,
            }
            _add_guidance(generator, gen_kwargs, request.model, guidance)

            result = generator.generate_image(**gen_kwargs)

            out_fmt = getattr(request, "output_format", "jpeg").upper()
            quality = getattr(request, "quality", 85) or 85

            if request.response_format == ResponseFormat.B64_JSON:
                b64 = _pil_to_b64(result.image, fmt=out_fmt, quality=quality)
                logger.info(f"Image {i+1} encoded: {len(b64)} chars b64 ({out_fmt} q={quality})")
                images.append(ImageObject(b64_json=b64, revised_prompt=request.prompt))
            else:
                ext = out_fmt.lower().replace("jpeg", "jpg")
                out_dir = Path(tempfile.gettempdir()) / "mlxgateway" / "images"
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / f"{int(time.time())}_{i}.{ext}"
                result.save(path=str(path), export_json_metadata=False)
                logger.info(f"Image {i+1} saved: {path}")
                images.append(ImageObject(url=f"file://{path}", revised_prompt=request.prompt))

        logger.info(f"Generated {len(images)} image(s)")
        return images

    def edit_images(self, request: ImageEditRequest) -> List[ImageObject]:
        params = request.get_extra_params()
        editor = _get_or_create_editor(request.model, params)

        if request.size:
            width, height = map(int, request.size.split("x"))
        else:
            from PIL import Image as PILImage
            first_img = PILImage.open(request.image_files[0])
            width, height = first_img.size
            logger.info(f"Auto-detected size: {width}x{height}")

        user_seed = params.get("seed")
        steps = params.get("steps", 4)
        guidance = params.get("guidance", 2.5)
        images: List[ImageObject] = []

        for i in range(request.n):
            seed = user_seed if user_seed is not None else random.randint(0, 2**32 - 1)

            logger.info(f"Editing image {i+1}/{request.n}: {width}x{height}, seed={seed}")

            gen_kwargs = {
                "seed": seed,
                "prompt": request.prompt,
                "image_paths": request.image_files,
                "num_inference_steps": steps,
                "width": width,
                "height": height,
            }
            _add_guidance(editor, gen_kwargs, request.model, guidance)

            result = editor.generate_image(**gen_kwargs)

            if request.response_format == ResponseFormat.B64_JSON:
                b64 = _pil_to_b64(result.image)
                images.append(ImageObject(b64_json=b64, revised_prompt=request.prompt))
            else:
                out_dir = Path(tempfile.gettempdir()) / "mlxgateway" / "images"
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / f"{int(time.time())}_{i}_edit.png"
                result.save(path=str(path), export_json_metadata=False)
                images.append(ImageObject(url=f"file://{path}", revised_prompt=request.prompt))

        logger.info(f"Edited {len(images)} image(s)")
        return images

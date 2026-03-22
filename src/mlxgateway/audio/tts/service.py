import gc
import inspect
import io
import threading
import time
import wave
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_audio.tts import load as load_tts_model
from mlx_audio.utils import load_audio

from ...utils.logger import logger
from .schema import AudioFormat, TTSRequest

_DEFAULT_REF_AUDIO_DIR = Path(__file__).resolve().parents[4] / "ref"

_SUPPORTED_FORMATS = {AudioFormat.WAV, AudioFormat.MP3, AudioFormat.FLAC, AudioFormat.PCM}

_tts_cache: OrderedDict[str, nn.Module] = OrderedDict()
_tts_cache_lock = threading.Lock()
_MAX_TTS_MODELS = 4


def _get_or_load_tts_model(model_id: str) -> nn.Module:
    with _tts_cache_lock:
        if model_id in _tts_cache:
            _tts_cache.move_to_end(model_id)
            logger.debug(f"TTS model cache hit: {model_id}")
            return _tts_cache[model_id]

    logger.info(f"Loading TTS model: {model_id}")
    model = load_tts_model(model_id)

    with _tts_cache_lock:
        if len(_tts_cache) >= _MAX_TTS_MODELS and model_id not in _tts_cache:
            oldest_key, _ = _tts_cache.popitem(last=False)
            mx.clear_cache()
            gc.collect()
            logger.info(f"TTS cache evicted: {oldest_key}")
        _tts_cache[model_id] = model

    return model


def _encode_wav(audio: mx.array, sample_rate: int) -> bytes:
    samples = np.array(audio.tolist(), dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _encode_audio(audio: mx.array, sample_rate: int, fmt: str) -> bytes:
    if fmt == "wav":
        return _encode_wav(audio, sample_rate)

    from mlx_audio.audio_io import write as audio_write
    buf = io.BytesIO()
    audio_write(buf, np.array(audio.tolist()), sample_rate, format=fmt)
    return buf.getvalue()


def _resolve_ref_audio_path(voice_name: str) -> Optional[Path]:
    """
    Resolve reference audio path from the reference directory based on the voice name.
    It looks for {voice_name}.* in the configured reference audio path or default ref dir.
    """
    from ...config import get_config
    config = get_config()
    
    # Determine the directory to search in
    if config.ref_audio_path:
        base_path = Path(config.ref_audio_path)
        # If it's explicitly a file that exists, and they aren't trying to dynamic route, 
        # we could just use it. But for dynamic voice routing, we treat it as a directory 
        # or get its parent if it's a file.
        if base_path.is_file():
            search_dir = base_path.parent
        else:
            search_dir = base_path
    else:
        search_dir = _DEFAULT_REF_AUDIO_DIR

    if not search_dir.exists() or not search_dir.is_dir():
        logger.warning(f"Reference audio directory does not exist: {search_dir}")
        return None

    # Supported extensions
    extensions = [".ogg", ".wav", ".mp3", ".flac"]
    
    # Clean the voice name to prevent directory traversal
    safe_voice = Path(voice_name).name
    
    for ext in extensions:
        potential_path = search_dir / f"{safe_voice}{ext}"
        if potential_path.exists() and potential_path.is_file():
            return potential_path
            
    # Fallback to default user if looking for something else failed? 
    # Or just return None so it falls back to model defaults.
    return None


class TTSService:
    def __init__(self, model: str):
        self.model = model

    def generate_speech_sync(self, request: TTSRequest) -> bytes:
        """Synchronous GPU inference - must be called from the MLX worker thread."""
        fmt = request.response_format or AudioFormat.WAV
        if fmt not in _SUPPORTED_FORMATS:
            raise ValueError(
                f"Audio format '{fmt.value}' is not supported. "
                f"Supported formats: {', '.join(f.value for f in _SUPPORTED_FORMATS)}"
            )

        logger.info(f"TTS request - model: {request.model}, voice: {request.voice}")
        model = _get_or_load_tts_model(self.model)

        instruct = (request.instruct or request.voice or "A clear neutral voice.").strip()
        voice = (request.voice or "af_sky").strip()

        gen_kwargs = {
            "text": request.input,
            "voice": voice,
            "speed": request.speed,
            "lang_code": "en",
            "verbose": False,
            "instruct": instruct,
        }
        gen_kwargs.update(request.get_extra_params() or {})

        if "ref_audio" in inspect.signature(model.generate).parameters:
            ref_path = _resolve_ref_audio_path(voice)
            if ref_path and "ref_audio" not in gen_kwargs:
                normalize = getattr(model, "model_type", "") == "spark"
                gen_kwargs["ref_audio"] = load_audio(
                    str(ref_path),
                    sample_rate=getattr(model, "sample_rate", 24000),
                    volume_normalize=normalize,
                )
                logger.info(f"Voice cloning with ref audio: {ref_path}")
            elif "ref_audio" not in gen_kwargs:
                logger.warning(f"Voice cloning requested for voice '{voice}' but no reference audio found.")

        logger.info(f"Generating audio - instruct: {instruct!r}, voice: {voice!r}")

        audio_chunks = []
        sample_rate = getattr(model, "sample_rate", 24000)
        for result in model.generate(**gen_kwargs):
            audio_chunks.append(result.audio)
            sample_rate = result.sample_rate

        if not audio_chunks:
            raise ValueError("Model returned no audio")

        audio = mx.concatenate(audio_chunks, axis=0) if len(audio_chunks) > 1 else audio_chunks[0]
        audio_bytes = _encode_audio(audio, sample_rate, fmt.value)

        logger.info(f"Audio generated: {len(audio_bytes)} bytes, sample_rate={sample_rate}")
        return audio_bytes

    async def generate_speech(self, request: TTSRequest) -> bytes:
        """Async entry point kept for backward compatibility."""
        from ...utils.gpu import run_on_mlx_thread
        return await run_on_mlx_thread(self.generate_speech_sync, request)

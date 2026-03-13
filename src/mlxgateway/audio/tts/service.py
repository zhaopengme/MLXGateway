import io
import threading
import time
import wave
from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_audio.tts import load as load_tts_model

from ...utils.logger import logger
from .schema import AudioFormat, TTSRequest

_SUPPORTED_FORMATS = {AudioFormat.WAV, AudioFormat.MP3, AudioFormat.FLAC, AudioFormat.PCM}

_tts_cache: Dict[str, nn.Module] = {}
_tts_cache_lock = threading.Lock()
_MAX_TTS_MODELS = 4


def _get_or_load_tts_model(model_id: str) -> nn.Module:
    with _tts_cache_lock:
        if model_id in _tts_cache:
            logger.debug(f"TTS model cache hit: {model_id}")
            return _tts_cache[model_id]

    logger.info(f"Loading TTS model: {model_id}")
    model = load_tts_model(model_id)

    with _tts_cache_lock:
        if len(_tts_cache) >= _MAX_TTS_MODELS and model_id not in _tts_cache:
            oldest_key = next(iter(_tts_cache))
            _tts_cache.pop(oldest_key)
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


class TTSService:
    def __init__(self, model: str):
        self.model = model

    async def generate_speech(self, request: TTSRequest) -> bytes:
        try:
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

        except Exception as e:
            logger.exception(f"TTS error: {e}")
            raise

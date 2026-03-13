import tempfile
import threading
from pathlib import Path
from typing import Dict, Union

import mlx.nn as nn
from mlx_audio.stt import load as load_stt_model
from mlx_audio.stt.generate import generate_transcription

from ...utils.logger import logger
from .schema import (
    ResponseFormat,
    STTRequestForm,
    TranscriptionResponse,
    TranscriptionWord,
)

_stt_cache: Dict[str, nn.Module] = {}
_stt_cache_lock = threading.Lock()
_MAX_STT_MODELS = 4


def _get_or_load_stt_model(model_id: str) -> nn.Module:
    with _stt_cache_lock:
        if model_id in _stt_cache:
            logger.debug(f"STT model cache hit: {model_id}")
            return _stt_cache[model_id]

    logger.info(f"Loading STT model: {model_id}")
    model = load_stt_model(model_id)

    with _stt_cache_lock:
        if len(_stt_cache) >= _MAX_STT_MODELS and model_id not in _stt_cache:
            oldest_key = next(iter(_stt_cache))
            _stt_cache.pop(oldest_key)
            logger.info(f"STT cache evicted: {oldest_key}")
        _stt_cache[model_id] = model

    return model


class STTService:
    async def _save_upload_file(self, file) -> str:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            return tmp.name

    def _format_response(
        self, result: dict, request: STTRequestForm
    ) -> Union[dict, str, TranscriptionResponse]:
        if request.response_format == ResponseFormat.TEXT:
            return result["text"]

        elif request.response_format == ResponseFormat.VERBOSE_JSON:
            return result

        elif request.response_format == ResponseFormat.JSON:
            return {"text": result["text"]}

        else:
            text = result.get("text", "")
            language = result.get("language", "en")

            duration = 0
            if "segments" in result:
                for segment in result["segments"]:
                    if "end" in segment:
                        duration = max(duration, segment["end"])

            words = []
            if request.timestamp_granularities and "word" in [
                g.value for g in request.timestamp_granularities
            ]:
                for segment in result.get("segments", []):
                    for word_data in segment.get("words", []):
                        word = TranscriptionWord(
                            word=word_data["word"],
                            start=word_data["start"],
                            end=word_data["end"],
                        )
                        words.append(word)

            return TranscriptionResponse(
                task="transcribe",
                language=language,
                duration=duration,
                text=text,
                words=words if words else None,
            )

    async def transcribe(
        self, request: STTRequestForm
    ) -> Union[dict, str, TranscriptionResponse]:
        audio_path = None
        try:
            logger.info(f"STT input - model: {request.model}, file: {request.file.filename}, language: {request.language}, temp: {request.temperature}")
            audio_path = await self._save_upload_file(request.file)

            model = _get_or_load_stt_model(request.model)

            logger.info(f"Transcribing audio: {audio_path}")
            gen_kwargs = {
                "temperature": request.temperature,
                "verbose": False,
            }
            if request.language is not None:
                gen_kwargs["language"] = request.language
            if request.prompt is not None:
                gen_kwargs["initial_prompt"] = request.prompt
            result = generate_transcription(
                model=model,
                audio=audio_path,
                **gen_kwargs,
            )

            result_dict = {
                "text": result.text,
                "language": result.language or "en",
                "segments": result.segments or [],
            }

            logger.info(f"STT output - text: {result.text}, language: {result.language or 'en'}, segments: {len(result.segments or [])}")

            response = self._format_response(result_dict, request)
            Path(audio_path).unlink(missing_ok=True)
            return response

        except Exception as e:
            if audio_path:
                Path(audio_path).unlink(missing_ok=True)
            raise e

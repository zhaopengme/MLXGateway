import json
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ..utils.logger import logger

# Truncate logged bodies larger than this (bytes).
_MAX_LOG_BODY_BYTES = 4096

# Maximum bytes to accumulate for stream response logging.
_MAX_STREAM_LOG_BYTES = 64 * 1024

# Paths whose response bodies are never logged (too large / binary).
_SKIP_RESPONSE_BODY_PREFIXES = ("/v1/embeddings", "/v1/images", "/v1/audio", "/v1/videos", "/static/")


def format_body(body: str, max_bytes: int = _MAX_LOG_BODY_BYTES) -> str:
    """Pretty-print JSON, truncating if the result exceeds max_bytes."""
    try:
        parsed = json.loads(body)
        formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        formatted = body

    if len(formatted.encode()) > max_bytes:
        truncated = formatted.encode()[:max_bytes].decode(errors="replace")
        return truncated + f"\n... [truncated, total {len(formatted)} chars]"
    return formatted


def _should_skip_response_body(path: str) -> bool:
    return any(path.startswith(p) for p in _SKIP_RESPONSE_BODY_PREFIXES)


class RequestResponseLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if _should_skip_response_body(request.url.path):
            return await call_next(request)

        body = await self._get_request_body(request)
        request_id = uuid.uuid4().hex[:8]

        logger.info(
            f"Request [{request_id}]: {request.method} {request.url}\n"
            f"Body:\n{format_body(body)}",
        )

        is_stream = False
        try:
            body_json = json.loads(body)
            if body_json.get("stream", False):
                is_stream = True
        except json.JSONDecodeError:
            pass

        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        skip_body = _should_skip_response_body(request.url.path)

        if is_stream:
            logger.info(
                f"First Stream Response [{request_id}] took {process_time:.2f}s:\n"
                f"Status: {response.status_code}\n"
            )

            if skip_body:
                return response

            async def stream_wrapper(iterator):
                full_body = b""
                truncated = False
                async for chunk in iterator:
                    if not truncated and len(full_body) < _MAX_STREAM_LOG_BYTES:
                        full_body += chunk
                        if len(full_body) >= _MAX_STREAM_LOG_BYTES:
                            truncated = True
                    yield chunk

                try:
                    body_text = full_body.decode()
                    formatted_text = ""
                    for line in body_text.splitlines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data != "[DONE]":
                                try:
                                    parsed = json.loads(data)
                                    formatted_text += json.dumps(parsed, indent=2, ensure_ascii=False) + "\n"
                                except json.JSONDecodeError:
                                    formatted_text += line + "\n"
                            else:
                                formatted_text += line + "\n"
                        elif line:
                            formatted_text += line + "\n"

                    suffix = " [truncated]" if truncated else ""
                    logger.info(
                        f"Stream Output Finished [{request_id}]{suffix}:\n"
                        f"{formatted_text.strip()}"
                    )
                except Exception as e:
                    logger.info(f"Stream Output Finished [{request_id}] (Parse error: {e})")

            response.body_iterator = stream_wrapper(response.body_iterator)
            return response

        # Non-streaming: buffer body only if we need to log it.
        if skip_body:
            logger.info(
                f"Response [{request_id}] took {process_time:.2f}s:\n"
                f"Status: {response.status_code} [body omitted for {request.url.path}]",
            )
            return response

        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk

        async def body_iterator():
            yield response_body

        response.body_iterator = body_iterator()

        try:
            body_text = response_body.decode()
            body_text = format_body(body_text)
        except UnicodeDecodeError:
            body_text = "<Binary Content>"

        logger.info(
            f"Response [{request_id}] took {process_time:.2f}s:\n"
            f"Status: {response.status_code}\n"
            f"Body:\n{body_text}",
        )
        return response

    async def _get_request_body(self, request: Request) -> str:
        try:
            body = await request.body()
            return body.decode()
        except Exception:
            return ""

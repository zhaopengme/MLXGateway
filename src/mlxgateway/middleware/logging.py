import json
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ..utils.logger import logger


def format_body(body: str) -> str:
    try:
        parsed = json.loads(body)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return body


class RequestResponseLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
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

        if is_stream:
            logger.info(
                f"First Stream Response [{request_id}] took {process_time:.2f}s:\n"
                f"Status: {response.status_code}\n"
            )

            # Wrapper for stream response
            async def stream_wrapper(iterator):
                full_body = b""
                async for chunk in iterator:
                    full_body += chunk
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
                            
                    logger.info(
                        f"Stream Output Finished [{request_id}]:\n"
                        f"{formatted_text.strip()}"
                    )
                except Exception as e:
                    logger.info(f"Stream Output Finished [{request_id}] (Parse error: {e})")

            response.body_iterator = stream_wrapper(response.body_iterator)
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

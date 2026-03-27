"""Cross-Origin-Resource-Policy for static assets.

Pages with Cross-Origin-Embedder-Policy: require-corp (e.g. FFmpeg.wasm) may only
embed cross-origin resources that opt in via CORP. Browser fetches to this gateway's
/static/* URLs need this header.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class StaticCrossOriginResourcePolicyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        return response

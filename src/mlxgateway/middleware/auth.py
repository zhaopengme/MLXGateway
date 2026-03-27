from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

_PUBLIC_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}
# Starlette normalizes path segments (resolves "..", decodes percent-encoding)
# before populating request.url.path, so prefix matching is safe here.
_PUBLIC_PREFIXES = ("/static/",)


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key: str):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in _PUBLIC_PATHS or any(path.startswith(p) for p in _PUBLIC_PREFIXES):
            return await call_next(request)

        # CORS preflight does not send Authorization; let inner middleware answer it.
        if request.method == "OPTIONS":
            return await call_next(request)

        auth = request.headers.get("Authorization", "")

        if auth.startswith("Bearer "):
            token = auth[7:]
        else:
            token = None

        if token != self.api_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Invalid API key.",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )

        return await call_next(request)

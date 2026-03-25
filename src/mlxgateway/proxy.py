"""Multi-process proxy dispatcher for MLXGateway.

Spawns 6 worker subprocesses (one per endpoint type), each listening on an
internal port. The main proxy process listens on the user-facing port and
forwards requests to the correct worker based on URL path.

This allows video generation (minutes) to never block embedding (milliseconds)
since they run in separate processes with independent Metal GPU contexts.
"""

import asyncio
import signal
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.staticfiles import StaticFiles

from .utils.static import STATIC_DIR, ensure_dirs
from .utils.logger import logger, set_logger_level

# Worker definitions: (name, routers, internal_port_offset)
WORKERS = [
    ("chat", "chat,models", 1),
    ("embedding", "embedding", 2),
    ("stt", "stt", 3),
    ("tts", "tts", 4),
    ("image", "image", 5),
    ("video", "video", 6),
]

# Path prefix -> worker name mapping
ROUTE_TABLE = [
    ("/v1/chat/", "chat"),
    ("/v1/models", "chat"),
    ("/v1/embeddings", "embedding"),
    ("/v1/audio/transcriptions", "stt"),
    ("/v1/audio/speech", "tts"),
    ("/v1/images/", "image"),
    ("/v1/videos/", "video"),
]


def _find_worker(path: str) -> str | None:
    """Match a request path to a worker name."""
    for prefix, name in ROUTE_TABLE:
        if path.startswith(prefix):
            return name
    return None


class WorkerManager:
    """Manages worker subprocess lifecycle."""

    def __init__(self, base_port: int, args):
        self.base_port = base_port
        self.args = args
        self.processes: dict[str, subprocess.Popen] = {}
        self.ports: dict[str, int] = {}
        self._http_client: httpx.AsyncClient | None = None

    def _build_worker_cmd(self, name: str, routers: str, port: int) -> list[str]:
        cmd = [
            sys.executable, "-m", "mlxgateway.main",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--routers", routers,
            "--log-level", self.args.log_level,
            "--model-cache-ttl", str(self.args.model_cache_ttl),
            "--max-models", str(self.args.max_models),
            "--max-concurrent", str(self.args.max_concurrent),
            "--request-timeout", str(self.args.request_timeout),
        ]
        # Don't pass --api-key to workers; auth is handled at the proxy layer only.
        if self.args.ref_audio:
            cmd.extend(["--ref-audio", self.args.ref_audio])
        cmd.extend(["--model-list-cache", str(self.args.model_list_cache)])
        return cmd

    def start_all(self):
        """Start all worker subprocesses."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        self._log_files: list = []

        for name, routers, offset in WORKERS:
            port = self.base_port + offset
            cmd = self._build_worker_cmd(name, routers, port)
            logger.info(f"Starting worker [{name}] on port {port}")
            log_file = open(log_dir / f"worker-{name}.log", "a")
            self._log_files.append(log_file)
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            self.processes[name] = proc
            self.ports[name] = port

    async def wait_healthy(self, timeout: float = 60):
        """Wait for all workers to respond to /health."""
        client = httpx.AsyncClient()
        deadline = time.monotonic() + timeout
        for name, port in self.ports.items():
            url = f"http://127.0.0.1:{port}/health"
            while time.monotonic() < deadline:
                try:
                    resp = await client.get(url, timeout=2)
                    if resp.status_code == 200:
                        logger.info(f"Worker [{name}] healthy on port {port}")
                        break
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass
                # Check if process died
                if self.processes[name].poll() is not None:
                    logger.error(f"Worker [{name}] exited with code {self.processes[name].returncode}")
                    break
                await asyncio.sleep(0.5)
            else:
                logger.warning(f"Worker [{name}] not healthy after {timeout}s")
        await client.aclose()

    async def get_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10))
        return self._http_client

    async def proxy_request(self, worker_name: str, request: Request):
        """Forward a request to the target worker and return the response."""
        port = self.ports[worker_name]
        target_url = f"http://127.0.0.1:{port}{request.url.path}"
        if request.url.query:
            target_url += f"?{request.url.query}"

        client = await self.get_client()

        headers = dict(request.headers)
        headers.pop("host", None)

        body = await request.body()

        try:
            # Peek at the response to determine if it's SSE streaming.
            # For SSE, we must keep the httpx stream open for the entire
            # duration of the response, so the generator owns the lifecycle.
            stream_ctx = client.stream(
                method=request.method,
                url=target_url,
                content=body,
                headers=headers,
            )
            resp = await stream_ctx.__aenter__()
            content_type = resp.headers.get("content-type", "")

            if "text/event-stream" in content_type:
                async def sse_gen():
                    try:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
                    finally:
                        await stream_ctx.__aexit__(None, None, None)

                return StreamingResponse(
                    sse_gen(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )

            # Non-streaming: read full body and close the stream
            try:
                resp_body = await resp.aread()
            finally:
                await stream_ctx.__aexit__(None, None, None)

            fwd_headers = {
                k: v for k, v in resp.headers.items()
                if k.lower() not in ("transfer-encoding", "connection", "keep-alive")
            }

            from fastapi.responses import Response
            return Response(
                content=resp_body,
                status_code=resp.status_code,
                headers=fwd_headers,
            )

        except httpx.TimeoutException:
            return JSONResponse(
                status_code=504,
                content={"error": {"message": f"Worker [{worker_name}] timed out", "type": "server_error", "code": "gateway_timeout"}},
            )
        except httpx.ConnectError:
            return JSONResponse(
                status_code=502,
                content={"error": {"message": f"Worker [{worker_name}] unavailable", "type": "server_error", "code": "bad_gateway"}},
            )

    def stop_all(self):
        """Stop all worker subprocesses."""
        for name, proc in self.processes.items():
            if proc.poll() is None:
                logger.info(f"Stopping worker [{name}] (PID {proc.pid})")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Worker [{name}] did not stop, killing")
                    proc.kill()

    def _restart_worker(self, name: str):
        """Restart a single crashed worker."""
        for wname, routers, offset in WORKERS:
            if wname == name:
                port = self.base_port + offset
                cmd = self._build_worker_cmd(name, routers, port)
                log_dir = Path("logs")
                log_file = open(log_dir / f"worker-{name}.log", "a")
                self._log_files.append(log_file)
                proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
                self.processes[name] = proc
                logger.info(f"Restarted worker [{name}] (PID {proc.pid}) on port {port}")
                break

    async def monitor_workers(self):
        """Background task: check worker health and restart crashed ones."""
        while True:
            await asyncio.sleep(5)
            for name, proc in list(self.processes.items()):
                if proc.poll() is not None:
                    logger.error(
                        f"Worker [{name}] died (exit={proc.returncode}), restarting..."
                    )
                    self._restart_worker(name)

    async def close(self):
        if self._http_client:
            await self._http_client.aclose()
        for f in getattr(self, "_log_files", []):
            try:
                f.close()
            except Exception:
                pass


def start_proxy(args):
    """Start the multi-process proxy dispatcher."""
    from .config import Config, set_config

    config = Config.from_env(
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        model_cache_ttl=args.model_cache_ttl,
        max_models=args.max_models,
        model_list_cache_ttl=args.model_list_cache,
        api_key=args.api_key,
        max_concurrent=args.max_concurrent,
        request_timeout=args.request_timeout,
        ref_audio_path=args.ref_audio,
    )
    set_config(config)
    set_logger_level(logger, config.log_level)

    # Internal base port = user port + 10000
    base_port = args.port + 10000
    manager = WorkerManager(base_port, args)

    @asynccontextmanager
    async def proxy_lifespan(app: FastAPI):
        manager.start_all()
        await manager.wait_healthy(timeout=120)
        logger.info(f"All workers started. Proxy listening on {args.host}:{args.port}")
        monitor_task = asyncio.create_task(manager.monitor_workers())
        yield
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        logger.info("Shutting down proxy and workers...")
        manager.stop_all()
        await manager.close()

    proxy_app = FastAPI(title="MLX Gateway (Multi-Process)", lifespan=proxy_lifespan)

    @proxy_app.get("/health")
    async def health():
        return {"status": "ok", "mode": "multi"}

    # Static files served directly by the proxy
    ensure_dirs()
    proxy_app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @proxy_app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def catch_all(request: Request, path: str):
        full_path = f"/{path}"
        worker_name = _find_worker(full_path)
        if worker_name is None:
            return JSONResponse(
                status_code=404,
                content={"error": {"message": f"No worker for path: {full_path}", "type": "invalid_request_error", "code": "not_found"}},
            )
        return await manager.proxy_request(worker_name, request)

    # Auth middleware on the proxy
    from .middleware.auth import APIKeyAuthMiddleware
    from fastapi.middleware.cors import CORSMiddleware

    proxy_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if config.api_key:
        proxy_app.add_middleware(APIKeyAuthMiddleware, api_key=config.api_key)

    logger.info(f"Starting MLX Gateway [multi-process] on {config.host}:{config.port}")
    logger.info(f"Workers: {', '.join(f'{name}:{base_port+offset}' for name, _, offset in WORKERS)}")

    uvicorn.run(
        proxy_app,
        host=config.host,
        port=config.port,
        log_level=config.log_level,
        use_colors=True,
    )

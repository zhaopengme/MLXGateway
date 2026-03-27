"""Shared static file output directories.

All generated files (images, videos) are stored under a single `static/`
directory at the project root, served by FastAPI's StaticFiles mount.
"""

import time
from pathlib import Path

# Anchor to the project root via __file__ so the path is deterministic
# regardless of the working directory when the process starts.
# __file__ -> utils/static.py -> mlxgateway/ -> src/ -> project root
STATIC_DIR = Path(__file__).resolve().parents[3] / "static"

IMAGES_DIR = STATIC_DIR / "images"
VIDEOS_DIR = STATIC_DIR / "videos"
TEMP_DIR = STATIC_DIR / ".tmp"


def ensure_dirs() -> None:
    """Create all static subdirectories. Called explicitly before StaticFiles mount."""
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)
    VIDEOS_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)


def get_base_url(request) -> str:
    """Return the public-facing base URL for static file links.

    In multi-process mode the proxy sets X-Forwarded-Host / X-Forwarded-Proto
    so that workers return URLs reachable by the client, not internal
    127.0.0.1:<worker_port> addresses.
    """
    fwd_host = request.headers.get("x-forwarded-host")
    if fwd_host:
        scheme = request.headers.get("x-forwarded-proto", "http")
        return f"{scheme}://{fwd_host}"
    return str(request.base_url).rstrip("/")


def cleanup_old_files(max_age_seconds: int = 3600) -> int:
    """Delete generated files older than max_age_seconds from output directories.

    Cleans IMAGES_DIR, VIDEOS_DIR, and TEMP_DIR. Returns the number of files removed.
    """
    removed = 0
    cutoff = time.time() - max_age_seconds
    for directory in (IMAGES_DIR, VIDEOS_DIR, TEMP_DIR):
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if path.is_file() and path.stat().st_mtime < cutoff:
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    pass
    return removed

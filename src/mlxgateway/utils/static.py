"""Shared static file output directories.

All generated files (images, videos) are stored under a single `static/`
directory at the project root, served by FastAPI's StaticFiles mount.
"""

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

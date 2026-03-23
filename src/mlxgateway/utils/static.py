"""Shared static file output directories.

All generated files (images, videos, audio) are stored under a single
`static/` directory at the project root, served by FastAPI's StaticFiles.
"""

import os
from pathlib import Path

# Project root: two levels up from this file (utils/ -> mlxgateway/ -> src/ -> project root)
# But we use CWD as the base, which is the project root when started via start.sh.
_BASE = Path(os.getcwd()) / "static"

IMAGES_DIR = _BASE / "images"
VIDEOS_DIR = _BASE / "videos"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

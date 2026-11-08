"""Where VideoRAG keeps its runtime data (all git-ignored)."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("VIDEORAG_DATA_DIR", os.path.join(ROOT, "data"))
VIDEO_DIR = os.path.join(DATA_DIR, "videos")   # uploaded videos, kept for playback
TEMP_DIR = os.path.join(DATA_DIR, "tmp")       # extracted audio, deleted after indexing
DB_PATH = os.path.join(DATA_DIR, "library.sqlite3")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")  # key-frame thumbnails (visual search only)

"""
visual.py
Optional visual indexing: what is *shown* in a lecture, not just what is said.

1. Sample a frame every few seconds (one ffmpeg pass) and keep only key
   frames, i.e. frames that differ visibly from the last kept one (a new
   slide, a new picture, the board after writing).
2. OCR each key frame for on-screen text (slides, board, code, captions).
3. Embed each key frame with CLIP so it can be matched by description
   ("a map of trade routes", "a circuit diagram").

Models are loaded lazily, so users who never enable visual search never
download or load them.
"""

import glob
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable

import numpy as np

from backend.processor import FFMPEG_CMD, ProcessingError, check_ffmpeg

logger = logging.getLogger(__name__)

CLIP_MODEL_NAME = "clip-ViT-B-32"
SAMPLE_SECONDS = 2.0          # look at one frame every N seconds
FRAME_WIDTH = 640             # analysis resolution (OCR is accurate enough at this size)
THUMB_WIDTH = 320             # stored preview shown in search results
CHANGE_THRESHOLD = 10.0       # mean abs pixel difference (0-255) that counts as a new key frame
MAX_KEYFRAME_GAP = 60.0       # keep a frame at least this often even if nothing changes
OCR_MIN_CONFIDENCE = 0.6

_model_lock = threading.Lock()
_ocr = None
_clip = None


def _get_ocr():
    global _ocr
    with _model_lock:
        if _ocr is None:
            from rapidocr import RapidOCR
            _ocr = RapidOCR()
    return _ocr


def _get_clip():
    global _clip
    with _model_lock:
        if _clip is None:
            from sentence_transformers import SentenceTransformer
            _clip = SentenceTransformer(CLIP_MODEL_NAME)
    return _clip


def embed_images(paths: list[str]) -> np.ndarray:
    """CLIP embeddings (L2-normalised, float32) for image files."""
    from PIL import Image
    images = [Image.open(p).convert("RGB") for p in paths]
    vecs = _get_clip().encode(images, normalize_embeddings=True, batch_size=16)
    return np.asarray(vecs, dtype=np.float32)


def embed_text(text: str) -> np.ndarray:
    """CLIP embedding of a text query, comparable with :func:`embed_images`."""
    vec = _get_clip().encode([text], normalize_embeddings=True)[0]
    return np.asarray(vec, dtype=np.float32)


# ---------------------------------------------------------------------------
# Key frames
# ---------------------------------------------------------------------------
def _signature(path: str) -> np.ndarray:
    """Tiny grayscale fingerprint used to detect visible change between frames."""
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize((48, 27)), dtype=np.float32)


def select_keyframes(
    signatures: list[np.ndarray],
    interval: float = SAMPLE_SECONDS,
    threshold: float = CHANGE_THRESHOLD,
    max_gap: float = MAX_KEYFRAME_GAP,
) -> list[int]:
    """Indices of sampled frames that start a new visual "scene"."""
    keep: list[int] = []
    for i, sig in enumerate(signatures):
        if not keep:
            keep.append(i)
            continue
        last = keep[-1]
        changed = float(np.abs(sig - signatures[last]).mean()) >= threshold
        if changed or (i - last) * interval >= max_gap:
            keep.append(i)
    return keep


def extract_keyframes(
    video_path: str,
    out_dir: str,
    duration: float,
    on_progress: Callable[[float], None] | None = None,
) -> list[dict]:
    """Sample the video and save its key frames as thumbnails in *out_dir*.

    Returns:
        ``[{start, end, path, thumb}]`` in time order, where ``path`` is a
        temporary full-size frame (for OCR/CLIP) and ``thumb`` the stored
        preview image.
    """
    check_ffmpeg()
    work = tempfile.mkdtemp(prefix="videorag-frames-")
    try:
        cmd = [FFMPEG_CMD, "-v", "error", "-y", "-i", video_path,
               "-vf", f"fps=1/{SAMPLE_SECONDS},scale={FRAME_WIDTH}:-2",
               "-q:v", "4", os.path.join(work, "s_%06d.jpg")]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            logger.warning("ffmpeg frame sampling failed: %s", exc.stderr)
            raise ProcessingError("Could not read the video frames.") from exc

        samples = sorted(glob.glob(os.path.join(work, "s_*.jpg")))
        if on_progress:
            on_progress(0.5)
        sigs = [_signature(p) for p in samples]
        keep = select_keyframes(sigs)

        os.makedirs(out_dir, exist_ok=True)
        frames: list[dict] = []
        from PIL import Image
        for n, i in enumerate(keep):
            start = i * SAMPLE_SECONDS
            nxt = keep[n + 1] * SAMPLE_SECONDS if n + 1 < len(keep) else max(duration, start + SAMPLE_SECONDS)
            thumb = os.path.join(out_dir, f"{n:05d}.jpg")
            with Image.open(samples[i]) as im:
                im.thumbnail((THUMB_WIDTH, THUMB_WIDTH))
                im.save(thumb, quality=80)
            frames.append({"start": round(start, 2), "end": round(nxt, 2),
                           "path": samples[i], "thumb": thumb})
        # Keep the work dir alive for OCR/CLIP; the caller cleans it up.
        for f in frames:
            f["workdir"] = work
        if on_progress:
            on_progress(1.0)
        return frames
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise


_LATIN = re.compile(r"[A-Za-z0-9]")


def clean_ocr_lines(lines: list[str], scores: list[float]) -> str:
    """Join confident OCR lines, dropping stray symbols from photos."""
    keep = []
    for text, score in zip(lines, scores):
        text = text.strip()
        if score >= OCR_MIN_CONFIDENCE and len(_LATIN.findall(text)) >= 2:
            keep.append(text)
    return " ".join(keep)


def read_text(path: str) -> str:
    """On-screen text in an image ('' when there is none)."""
    result = _get_ocr()(path)
    if not result or not result.txts:
        return ""
    return clean_ocr_lines(list(result.txts), [float(s) for s in result.scores])


def analyse_video(
    video_path: str,
    thumbs_dir: str,
    duration: float,
    report: Callable[[str, float], None],
) -> list[dict]:
    """Key frames with their OCR text and CLIP embedding.

    Returns:
        ``[{start, end, thumb, text, embedding}]``.
    """
    report("Finding key frames", 0.0)
    frames = extract_keyframes(video_path, thumbs_dir, duration,
                               on_progress=lambda p: report("Finding key frames", p))
    if not frames:
        return []
    work = frames[0]["workdir"]
    try:
        for n, f in enumerate(frames):
            f["text"] = read_text(f["path"])
            report("Reading slides and on-screen text", (n + 1) / len(frames))
        report("Understanding images", 0.0)
        embeddings = []
        batch = 32
        for i in range(0, len(frames), batch):
            embeddings.extend(embed_images([f["path"] for f in frames[i:i + batch]]))
            report("Understanding images", min(1.0, (i + batch) / len(frames)))
        for f, emb in zip(frames, embeddings):
            f["embedding"] = emb
        return [{k: f[k] for k in ("start", "end", "thumb", "text", "embedding")} for f in frames]
    finally:
        shutil.rmtree(work, ignore_errors=True)

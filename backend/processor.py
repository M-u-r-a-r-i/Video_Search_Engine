"""
processor.py
Audio extraction (ffmpeg) and speech-to-text transcription (faster-whisper).
"""

import os
import glob
import shutil
import sys
import logging
import ffmpeg
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp_data")
WHISPER_MODEL_SIZE = "base"

os.makedirs(TEMP_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Whisper model (loaded once per process, device auto-detected)
# ---------------------------------------------------------------------------
_whisper_model: WhisperModel | None = None


def _select_device() -> tuple[str, str]:
    """Return ``(device, compute_type)`` based on CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:  # torch missing or broken — fall back to CPU
        logger.debug("CUDA probe failed; using CPU", exc_info=True)
    return "cpu", "int8"


def _get_whisper_model() -> WhisperModel:
    """Lazily load and cache the Whisper model as a process singleton."""
    global _whisper_model
    if _whisper_model is None:
        device, compute_type = _select_device()
        try:
            _whisper_model = WhisperModel(
                WHISPER_MODEL_SIZE, device=device, compute_type=compute_type
            )
            logger.info("Loaded Whisper '%s' on %s (%s)",
                        WHISPER_MODEL_SIZE, device, compute_type)
        except (RuntimeError, ValueError):
            # GPU libs missing / unsupported compute type — fall back to CPU
            logger.warning("Whisper load on %s failed; falling back to CPU",
                           device, exc_info=True)
            _whisper_model = WhisperModel(
                WHISPER_MODEL_SIZE, device="cpu", compute_type="int8"
            )
    return _whisper_model


# ---------------------------------------------------------------------------
# Resolve ffmpeg binary (Windows PATH may not be refreshed in this process)
# ---------------------------------------------------------------------------
def _find_ffmpeg() -> str:
    """Return the absolute path to the ffmpeg executable."""
    found = shutil.which("ffmpeg")
    if found:
        return found

    # On Windows the current process may have inherited a stale PATH.
    # Read the *live* User + Machine PATH from the registry and search there.
    if sys.platform == "win32":
        machine = os.environ.get("Path", "")
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as key:
                machine = winreg.QueryValueEx(key, "Path")[0]
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
                user = winreg.QueryValueEx(key, "Path")[0]
            live_path = f"{machine};{user}"
            found = shutil.which("ffmpeg", path=live_path)
            if found:
                return found
        except OSError:
            logger.debug("Registry PATH lookup for ffmpeg failed", exc_info=True)

    return "ffmpeg"  # last resort — let subprocess raise if missing


FFMPEG_CMD = _find_ffmpeg()

FRAME_INTERVAL = 5  # extract one frame every N seconds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_audio(video_path: str) -> str:
    """Extract mono 16 kHz WAV from a video file.

    Args:
        video_path: Absolute path to the input video.

    Returns:
        Path to the extracted .wav file inside ``temp_data/``.
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(TEMP_DIR, f"{base_name}.wav")

    # Overwrite if a previous extraction exists
    if os.path.exists(audio_path):
        os.remove(audio_path)

    (
        ffmpeg
        .input(video_path)
        .output(audio_path, ac=1, ar=16000, format="wav")
        .overwrite_output()
        .run(cmd=FFMPEG_CMD, quiet=True)
    )

    return audio_path


def transcribe_audio(audio_path: str) -> list[dict]:
    """Transcribe an audio file using faster-whisper.

    Args:
        audio_path: Path to a WAV file (mono, 16 kHz recommended).

    Returns:
        List of segment dicts, each with keys ``start`` (float, seconds),
        ``end`` (float, seconds), and ``text`` (str).
    """
    model = _get_whisper_model()
    segments_gen, _info = model.transcribe(audio_path, vad_filter=True)

    segments: list[dict] = []
    for seg in segments_gen:
        segments.append(
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            }
        )

    return segments


def extract_frames(
    video_path: str,
    video_id: str,
    interval: float = FRAME_INTERVAL,
) -> list[dict]:
    """Extract frames from a video at regular intervals in a single ffmpeg pass.

    Args:
        video_path: Absolute path to the input video.
        video_id:   Unique identifier for the video (used in output path).
        interval:   Seconds between extracted frames (default 5).

    Returns:
        List of ``{timestamp: float, frame_path: str}`` dicts.
    """
    frames_dir = os.path.join(TEMP_DIR, "frames", video_id)
    os.makedirs(frames_dir, exist_ok=True)

    # Clear any stale frames from a previous run so indices stay aligned.
    for stale in glob.glob(os.path.join(frames_dir, "frame_*.jpg")):
        try:
            os.remove(stale)
        except OSError:
            logger.debug("Could not remove stale frame %s", stale, exc_info=True)

    out_pattern = os.path.join(frames_dir, "frame_%05d.jpg")

    # One ffmpeg invocation extracts every frame via the fps filter — far
    # faster than spawning a separate process per timestamp.
    try:
        (
            ffmpeg
            .input(video_path)
            .output(out_pattern, vf=f"fps=1/{interval}", **{"q:v": 2})
            .overwrite_output()
            .run(cmd=FFMPEG_CMD, quiet=True)
        )
    except Exception:
        logger.warning("Frame extraction failed for %s", video_path, exc_info=True)
        return []

    # The fps filter emits one frame per `interval` seconds starting at t=0,
    # so frame N (0-indexed) corresponds to timestamp N * interval.
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    frames: list[dict] = []
    for idx, path in enumerate(frame_files):
        if os.path.getsize(path) > 0:
            frames.append({"timestamp": round(idx * interval, 2), "frame_path": path})

    return frames


def cleanup_artifacts(video_path: str, video_id: str) -> None:
    """Remove intermediate files (extracted WAV + frames) after indexing.

    The source video is preserved (it is still needed for playback); only the
    derived artifacts, which live in ``temp_data/`` and are no longer required
    once embeddings are stored, are deleted.

    Args:
        video_path: Path to the source video (used to locate its WAV).
        video_id:   Video identifier (used to locate its frames directory).
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(TEMP_DIR, f"{base_name}.wav")
    if os.path.exists(audio_path):
        try:
            os.remove(audio_path)
        except OSError:
            logger.debug("Could not remove %s", audio_path, exc_info=True)

    frames_dir = os.path.join(TEMP_DIR, "frames", video_id)
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir, ignore_errors=True)

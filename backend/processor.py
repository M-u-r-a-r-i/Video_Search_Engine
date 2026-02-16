"""
processor.py
Audio extraction (ffmpeg) and speech-to-text transcription (faster-whisper).
"""

import os
import shutil
import sys
import ffmpeg
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp_data")
WHISPER_MODEL_SIZE = "base"
WHISPER_COMPUTE_TYPE = "int8"

os.makedirs(TEMP_DIR, exist_ok=True)


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
            pass

    return "ffmpeg"  # last resort — let subprocess raise if missing


FFMPEG_CMD = _find_ffmpeg()

FRAME_INTERVAL = 5  # extract one frame every N seconds


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_video_duration(video_path: str) -> float:
    """Probe the duration of a video file in seconds."""
    try:
        probe = ffmpeg.probe(video_path, cmd=FFMPEG_CMD.replace("ffmpeg", "ffprobe"))
        return float(probe["format"]["duration"])
    except Exception:
        return 0.0


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
    try:
        model = WhisperModel(WHISPER_MODEL_SIZE, compute_type=WHISPER_COMPUTE_TYPE)
        segments_gen, _info = model.transcribe(audio_path, vad_filter=True)
    except RuntimeError:
        # CUDA libs missing — fall back to CPU
        model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type=WHISPER_COMPUTE_TYPE)
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
    """Extract frames from a video at regular intervals.

    Args:
        video_path: Absolute path to the input video.
        video_id:   Unique identifier for the video (used in output path).
        interval:   Seconds between extracted frames (default 5).

    Returns:
        List of ``{timestamp: float, frame_path: str}`` dicts.
    """
    frames_dir = os.path.join(TEMP_DIR, "frames", video_id)
    os.makedirs(frames_dir, exist_ok=True)

    duration = _get_video_duration(video_path)
    if duration <= 0:
        return []

    frames: list[dict] = []
    timestamp = 0.0
    idx = 0

    while timestamp < duration:
        out_path = os.path.join(frames_dir, f"frame_{idx:05d}.jpg")
        try:
            (
                ffmpeg
                .input(video_path, ss=timestamp)
                .output(out_path, vframes=1, format="image2", **{"q:v": 2})
                .overwrite_output()
                .run(cmd=FFMPEG_CMD, quiet=True)
            )
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                frames.append({"timestamp": round(timestamp, 2), "frame_path": out_path})
        except Exception:
            pass  # skip frame on error

        timestamp += interval
        idx += 1

    return frames

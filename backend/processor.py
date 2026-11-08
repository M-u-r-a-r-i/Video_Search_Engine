"""
processor.py
Audio extraction (ffmpeg) and speech-to-text with word-level timestamps
(faster-whisper), turned into timestamped sentences.
"""

import os
import shutil
import subprocess
import sys
import logging
from collections.abc import Callable

from faster_whisper import WhisperModel

from backend.paths import TEMP_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WHISPER_MODEL_SIZE = "base"

# A "sentence" longer than this is split at a word boundary. Whisper sometimes
# emits long runs without punctuation; without a cap one result could span
# minutes and the jump-to time would be useless.
MAX_SENTENCE_SECONDS = 15.0
MAX_SENTENCE_WORDS = 40
SENTENCE_END = (".", "?", "!")


class ProcessingError(Exception):
    """A user-facing problem with the input video or the environment."""


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
# Resolve ffmpeg / ffprobe (Windows PATH may not be refreshed in this process)
# ---------------------------------------------------------------------------
def _find_tool(name: str) -> str | None:
    """Return the absolute path to an ffmpeg-suite executable, or None."""
    found = shutil.which(name)
    if found:
        return found

    # On Windows the current process may have inherited a stale PATH.
    # Read the *live* User + Machine PATH from the registry and search there.
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as key:
                machine = winreg.QueryValueEx(key, "Path")[0]
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
                user = winreg.QueryValueEx(key, "Path")[0]
            return shutil.which(name, path=f"{machine};{user}")
        except OSError:
            logger.debug("Registry PATH lookup for %s failed", name, exc_info=True)
    return None


FFMPEG_CMD = _find_tool("ffmpeg")
FFPROBE_CMD = _find_tool("ffprobe")


def check_ffmpeg() -> None:
    """Raise :class:`ProcessingError` if ffmpeg/ffprobe are not installed."""
    if not FFMPEG_CMD or not FFPROBE_CMD:
        raise ProcessingError(
            "FFmpeg was not found. Install it (see the README) and restart the app."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def probe_video(video_path: str) -> dict:
    """Return ``{duration: float, has_audio: bool}`` for a video file."""
    check_ffmpeg()
    try:
        out = subprocess.run(
            [FFPROBE_CMD, "-v", "error",
             "-show_entries", "format=duration:stream=codec_type",
             "-of", "default=noprint_wrappers=1", video_path],
            capture_output=True, text=True, check=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise ProcessingError(
            "This file could not be read as a video. It may be corrupt or "
            "in an unsupported format."
        ) from exc

    duration = 0.0
    for line in out.splitlines():
        if line.startswith("duration="):
            try:
                duration = float(line.split("=", 1)[1])
            except ValueError:
                pass
    return {"duration": duration, "has_audio": "codec_type=audio" in out}


def extract_audio(video_path: str, video_id: str) -> str:
    """Extract mono 16 kHz WAV from a video file.

    Returns:
        Path to the extracted .wav file inside ``temp_data/``.

    Raises:
        ProcessingError: if ffmpeg is missing, the video has no audio track,
            or extraction fails.
    """
    info = probe_video(video_path)
    if not info["has_audio"]:
        raise ProcessingError(
            "This video has no audio track, so there is no speech to search."
        )

    os.makedirs(TEMP_DIR, exist_ok=True)
    audio_path = os.path.join(TEMP_DIR, f"{video_id}.wav")
    try:
        subprocess.run(
            [FFMPEG_CMD, "-y", "-v", "error", "-i", video_path,
             "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", audio_path],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.warning("ffmpeg audio extraction failed: %s", exc.stderr)
        raise ProcessingError("Could not extract audio from this video.") from exc
    return audio_path


def transcribe_audio(
    audio_path: str,
    duration: float = 0.0,
    on_progress: Callable[[float], None] | None = None,
) -> list[dict]:
    """Transcribe audio into timestamped sentences.

    Args:
        audio_path:  Path to a mono 16 kHz WAV.
        duration:    Audio length in seconds (only used for progress).
        on_progress: Optional callback receiving progress in ``[0, 1]``.

    Returns:
        List of ``{start, end, text}`` sentence dicts, in time order. Times
        come from Whisper's word timestamps, so ``start`` is the moment the
        first word of the sentence is spoken.
    """
    model = _get_whisper_model()
    segments, _info = model.transcribe(
        audio_path, vad_filter=True, word_timestamps=True
    )

    words: list[dict] = []
    for seg in segments:
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                words.append({"start": w.start, "end": w.end, "text": text})
        if on_progress and duration > 0:
            on_progress(min(seg.end / duration, 1.0))

    return words_to_sentences(words)


def words_to_sentences(
    words: list[dict],
    max_seconds: float = MAX_SENTENCE_SECONDS,
    max_words: int = MAX_SENTENCE_WORDS,
) -> list[dict]:
    """Group timestamped words into sentences.

    A sentence ends at ``.``, ``?`` or ``!``, or is force-split once it
    exceeds *max_seconds* or *max_words* (unpunctuated speech).
    """
    sentences: list[dict] = []
    current: list[dict] = []

    def flush() -> None:
        if current:
            sentences.append({
                "start": round(current[0]["start"], 2),
                "end": round(current[-1]["end"], 2),
                "text": " ".join(w["text"] for w in current),
            })
            current.clear()

    for w in words:
        current.append(w)
        too_long = (w["end"] - current[0]["start"] >= max_seconds
                    or len(current) >= max_words)
        if w["text"].endswith(SENTENCE_END) or too_long:
            flush()
    flush()
    return sentences


def cleanup_audio(video_id: str) -> None:
    """Remove the extracted WAV once the video is indexed."""
    audio_path = os.path.join(TEMP_DIR, f"{video_id}.wav")
    try:
        os.remove(audio_path)
    except FileNotFoundError:
        pass
    except OSError:
        logger.debug("Could not remove %s", audio_path, exc_info=True)

"""
jobs.py
Background processing: videos are transcribed and indexed one at a time on a
worker thread, while the website polls their progress.
"""

import logging
import os
import queue
import shutil
import threading
import time
from collections.abc import Callable

from backend import processor, store
from backend.paths import FRAMES_DIR

logger = logging.getLogger(__name__)

Reporter = Callable[[str, float], None]   # (stage label, progress 0..1)


def process_video(video_id: str, path: str, name: str, report: Reporter, *,
                  audio: bool = True, visual: bool = False) -> None:
    """Pipeline for one video.

    audio:  probe → extract audio → transcribe → index speech. After this
            step the lecture is searchable, even if visuals are still running.
    visual: key frames → on-screen text (OCR) + picture embeddings (CLIP).
            Optional and opt-in, because it is slower and needs extra models.
    """
    if audio:
        try:
            info = processor.probe_video(path)
            report("Extracting audio", 0.0)
            audio_path = processor.extract_audio(path, video_id)
            report("Transcribing speech", 0.0)
            sentences = processor.transcribe_audio(
                audio_path, info["duration"],
                on_progress=lambda p: report("Transcribing speech", p),
            )
            report("Building search index", 1.0)
            store.index_video(video_id, name, path, info["duration"], sentences)
        finally:
            processor.cleanup_audio(video_id)
    if visual:
        from backend import visual as vis
        video = store.get_video(video_id)
        duration = video["duration"] if video else processor.probe_video(path)["duration"]
        thumbs = os.path.join(FRAMES_DIR, video_id)
        shutil.rmtree(thumbs, ignore_errors=True)
        try:
            frames = vis.analyse_video(path, thumbs, duration, report)
            report("Building visual index", 1.0)
            store.index_frames(video_id, frames)
        except BaseException:
            shutil.rmtree(thumbs, ignore_errors=True)
            raise


class JobManager:
    """Runs processing jobs sequentially on one daemon thread.

    Whisper is CPU/GPU heavy, so running videos one after another is faster
    overall than running them in parallel. State lives in memory: a job that
    finished is visible in the store; a job lost to a restart is simply
    uploaded again.
    """

    def __init__(self, pipeline: Callable[..., None] = process_video):
        self._pipeline = pipeline
        self._queue: queue.Queue[str] = queue.Queue()
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="videorag-worker", daemon=True)
        self._thread.start()

    def submit(self, video_id: str, path: str, name: str, *,
               audio: bool = True, visual: bool = False) -> dict:
        """Queue a video unless it is already queued or processing.

        ``audio=False, visual=True`` adds visual search to an indexed video.
        """
        with self._lock:
            job = self._jobs.get(video_id)
            if job and job["status"] in ("queued", "processing"):
                return dict(job)
            job = {
                "video_id": video_id, "name": name, "path": path,
                "status": "queued", "stage": "Waiting in queue", "progress": 0.0,
                "error": None, "submitted_at": time.time(),
                "audio": audio, "visual": visual,
            }
            self._jobs[video_id] = job
        self._queue.put(video_id)
        return dict(job)

    def get(self, video_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(video_id)
            return dict(job) if job else None

    def active(self) -> list[dict]:
        """Jobs that are not finished successfully (queued, processing, failed)."""
        with self._lock:
            return [dict(j) for j in self._jobs.values() if j["status"] != "ready"]

    def forget(self, video_id: str) -> None:
        with self._lock:
            job = self._jobs.get(video_id)
            if job and job["status"] not in ("queued", "processing"):
                del self._jobs[video_id]

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Block until every queued job has finished (used by tests)."""
        end = time.time() + timeout
        while time.time() < end:
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.02)
        return False

    # ------------------------------------------------------------------ worker
    def _update(self, video_id: str, **fields) -> None:
        with self._lock:
            self._jobs[video_id].update(fields)

    def _run(self) -> None:
        while True:
            video_id = self._queue.get()
            try:
                job = self.get(video_id)
                self._update(video_id, status="processing", stage="Starting", progress=0.0)

                def report(stage: str, progress: float, _vid=video_id) -> None:
                    self._update(_vid, stage=stage, progress=round(progress, 3))

                self._pipeline(video_id, job["path"], job["name"], report,
                               audio=job["audio"], visual=job["visual"])
                self._update(video_id, status="ready", stage="Done", progress=1.0)
            except processor.ProcessingError as exc:
                self._update(video_id, status="error", error=str(exc))
            except Exception:
                logger.exception("Processing failed for %s", video_id)
                self._update(video_id, status="error",
                             error="Something went wrong while processing this video. "
                                   "Details are in the terminal running VideoRAG.")
            finally:
                self._queue.task_done()

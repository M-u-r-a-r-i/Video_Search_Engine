"""
api.py
The VideoRAG web server: a small JSON API plus the static website.

    GET    /                          the website
    GET    /api/videos                library (indexed + processing + failed)
    POST   /api/videos?filename=...   upload a video (raw request body)
    GET    /api/videos/{id}           one video and its processing status
    GET    /api/videos/{id}/file      the video itself (supports seeking)
    GET    /api/videos/{id}/search?q= search what is said in the video
    DELETE /api/videos/{id}           remove a video and its index
"""

import hashlib
import mimetypes
import os
import re
import uuid

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend import store
from backend.jobs import JobManager
from backend.paths import ROOT, VIDEO_DIR

WEB_DIR = os.path.join(ROOT, "web")
ALLOWED_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}
VIDEO_ID = re.compile(r"^[0-9a-f]{16}$")
UPLOAD_CHUNK = 1024 * 1024


def _check_id(video_id: str) -> None:
    if not VIDEO_ID.match(video_id):
        raise HTTPException(404, "Video not found")


def create_app(jobs: JobManager | None = None) -> FastAPI:
    app = FastAPI(title="VideoRAG", docs_url="/api/docs", redoc_url=None)
    app.state.jobs = jobs or JobManager()

    def _status(video_id: str) -> dict | None:
        """Merge what the index knows with what the job queue knows."""
        video = store.get_video(video_id)
        if video:
            return {
                "id": video_id, "name": video["name"], "duration": video["duration"],
                "status": "ready", "has_speech": video["chunks"] > 0,
                "indexed_at": video["indexed_at"],
            }
        job = app.state.jobs.get(video_id)
        if job:
            return {
                "id": video_id, "name": job["name"], "status": job["status"],
                "stage": job["stage"], "progress": job["progress"], "error": job["error"],
                "submitted_at": job["submitted_at"],
            }
        return None

    # ------------------------------------------------------------------ library
    @app.get("/api/videos")
    def list_videos() -> list[dict]:
        ready = [_status(v["id"]) for v in store.list_videos()]
        ready_ids = {v["id"] for v in ready}
        pending = [_status(j["video_id"]) for j in app.state.jobs.active()
                   if j["video_id"] not in ready_ids]
        pending.sort(key=lambda v: v["submitted_at"], reverse=True)
        return pending + ready

    @app.get("/api/videos/{video_id}")
    def get_video(video_id: str) -> dict:
        _check_id(video_id)
        status = _status(video_id)
        if status is None:
            raise HTTPException(404, "Video not found")
        return status

    @app.post("/api/videos", status_code=201)
    async def upload(request: Request, filename: str = Query(..., min_length=1)) -> dict:
        """Upload a video as the raw request body; processing starts automatically.

        The file is named by its content hash, so uploading the same lecture
        again (even renamed) reuses the existing index instantly.
        """
        name = os.path.basename(filename)
        ext = os.path.splitext(name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(400, f"Unsupported file type '{ext or name}'. "
                                     f"Use one of: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

        os.makedirs(VIDEO_DIR, exist_ok=True)
        tmp_path = os.path.join(VIDEO_DIR, f".upload-{uuid.uuid4().hex}{ext}")
        digest = hashlib.md5()
        size = 0
        try:
            with open(tmp_path, "wb") as f:
                async for chunk in request.stream():
                    digest.update(chunk)
                    f.write(chunk)
                    size += len(chunk)
        except BaseException:
            _remove(tmp_path)
            raise
        if size == 0:
            _remove(tmp_path)
            raise HTTPException(400, "The uploaded file is empty.")

        video_id = digest.hexdigest()[:16]
        path = os.path.join(VIDEO_DIR, video_id + ext)
        if os.path.exists(path):
            _remove(tmp_path)
        else:
            os.replace(tmp_path, path)

        if not store.is_indexed(video_id):
            app.state.jobs.submit(video_id, path, name)
        return _status(video_id)

    @app.delete("/api/videos/{video_id}", status_code=204)
    def delete(video_id: str) -> None:
        _check_id(video_id)
        job = app.state.jobs.get(video_id)
        if job and job["status"] in ("queued", "processing"):
            raise HTTPException(409, "This video is still being processed.")
        video = store.get_video(video_id)
        if video is None and job is None:
            raise HTTPException(404, "Video not found")
        path = video["path"] if video else job["path"]
        store.delete_video(video_id)
        app.state.jobs.forget(video_id)
        _remove(path)

    # ------------------------------------------------------------------ playback
    @app.get("/api/videos/{video_id}/file")
    def video_file(video_id: str) -> FileResponse:
        _check_id(video_id)
        video = store.get_video(video_id)
        job = app.state.jobs.get(video_id)
        path = video["path"] if video else (job["path"] if job else None)
        if not path or not os.path.exists(path):
            raise HTTPException(404, "Video file not found")
        media_type = mimetypes.guess_type(path)[0] or "video/mp4"
        return FileResponse(path, media_type=media_type)   # handles Range requests (seeking)

    # ------------------------------------------------------------------ search
    @app.get("/api/videos/{video_id}/search")
    def search(video_id: str, q: str = Query(..., min_length=1, max_length=500),
               k: int = Query(5, ge=1, le=20)) -> dict:
        _check_id(video_id)
        if not store.is_indexed(video_id):
            raise HTTPException(409, "This video is not ready to search yet.")
        return {"query": q, "topic": store.is_topic_query(q),
                "results": store.search(video_id, q, top_k=k)}

    # ------------------------------------------------------------------ website
    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(os.path.join(WEB_DIR, "index.html"))

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass

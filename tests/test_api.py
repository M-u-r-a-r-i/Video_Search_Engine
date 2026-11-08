import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend.api as api
import backend.store as store
from backend.jobs import JobManager
from backend.processor import ProcessingError
from test_store import LECTURE, _fake_embed, _fake_rerank


def _unit(i, dim=512):
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


# Key frames of the fake lecture: two slides with text, then pictures.
FAKE_FRAMES = [
    {"start": 0.0, "end": 20.0, "text": "Welcome to Optimisation 101", "embedding": _unit(0)},
    {"start": 20.0, "end": 45.0, "text": "Gradient Descent: w = w - lr * grad", "embedding": _unit(1)},
] + [
    {"start": 45.0 + 25 * k, "end": 70.0 + 25 * k, "text": "", "embedding": _unit(2 + k)}
    for k in range(6)
]
CAT_FRAME = 7   # the last frame "shows a cat"

PIPELINE_CALLS = []


def fake_pipeline(video_id, path, name, report, audio=True, visual=False):
    """Stands in for ffmpeg + Whisper (+ OCR/CLIP): indexes canned data."""
    PIPELINE_CALLS.append({"video_id": video_id, "audio": audio, "visual": visual})
    with open(path, "rb") as f:
        content = f.read()
    if audio:
        report("Transcribing speech", 0.5)
        if content.startswith(b"NOAUDIO"):
            raise ProcessingError("This video has no audio track, so there is no speech to search.")
        if content.startswith(b"CRASH"):
            raise RuntimeError("boom")
        sentences = [] if content.startswith(b"SILENT") else LECTURE
        store.index_video(video_id, name, path, 60.0, sentences)
    if visual:
        report("Reading slides and on-screen text", 0.5)
        if content.startswith(b"VISFAIL"):
            raise ProcessingError("Could not read the video frames.")
        thumbs = os.path.join(api.FRAMES_DIR, video_id)
        os.makedirs(thumbs, exist_ok=True)
        for i in range(len(FAKE_FRAMES)):
            Path(thumbs, f"{i:05d}.jpg").write_bytes(b"fake-jpeg")
        store.index_frames(video_id, FAKE_FRAMES)


def fake_clip_text(query):
    """'cat' queries point at the cat frame; anything else matches nothing in particular."""
    if "cat" in query.lower():
        return _unit(2 + 5)
    v = np.full(512, 0.01, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(store, "DB_PATH", str(tmp_path / "test.sqlite3"))
    monkeypatch.setattr(store, "_embed", _fake_embed)
    monkeypatch.setattr(store, "_rerank", _fake_rerank)
    monkeypatch.setattr(api, "VIDEO_DIR", str(tmp_path / "videos"))
    monkeypatch.setattr(api, "FRAMES_DIR", str(tmp_path / "frames"))
    monkeypatch.setattr(store, "_clip_text", fake_clip_text)
    store._emb_cache.clear()
    store._frame_cache.clear()
    PIPELINE_CALLS.clear()
    jobs = JobManager(pipeline=fake_pipeline)
    with TestClient(api.create_app(jobs)) as c:
        c.jobs = jobs
        yield c


def upload(client, content=b"VIDEO-BYTES-1", filename="Lecture 1.mp4", visual=False):
    params = {"filename": filename}
    if visual:
        params["visual"] = "true"
    return client.post("/api/videos", params=params, content=content)


def test_website_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "VideoRAG" in r.text
    assert client.get("/static/app.js").status_code == 200


def test_upload_processes_in_background_then_searches(client):
    r = upload(client)
    assert r.status_code == 201
    vid = r.json()["id"]
    assert r.json()["status"] in ("queued", "processing", "ready")

    assert client.jobs.wait_idle()
    v = client.get(f"/api/videos/{vid}").json()
    assert v["status"] == "ready" and v["has_speech"] and v["name"] == "Lecture 1.mp4"

    res = client.get(f"/api/videos/{vid}/search", params={"q": "what controls the learning rate step size"}).json()
    assert res["topic"] is False
    assert res["results"][0]["sentence"] == "The learning rate controls the step size."


def test_same_video_uploaded_again_is_instant_and_not_duplicated(client):
    vid = upload(client, filename="a.mp4").json()["id"]
    client.jobs.wait_idle()
    again = upload(client, filename="renamed copy.mp4")
    assert again.json()["id"] == vid and again.json()["status"] == "ready"
    assert len(client.get("/api/videos").json()) == 1


def test_different_videos_with_same_name_do_not_collide(client):
    a = upload(client, b"AAA", "lecture.mp4").json()["id"]
    b = upload(client, b"BBB", "lecture.mp4").json()["id"]
    client.jobs.wait_idle()
    assert a != b
    assert {v["id"] for v in client.get("/api/videos").json()} == {a, b}


def test_processing_errors_are_reported_to_the_user(client):
    vid = upload(client, b"NOAUDIO...", "silent-film.mp4").json()["id"]
    client.jobs.wait_idle()
    v = client.get(f"/api/videos/{vid}").json()
    assert v["status"] == "error" and "no audio track" in v["error"]

    vid2 = upload(client, b"CRASH...", "broken.mp4").json()["id"]
    client.jobs.wait_idle()
    assert "Something went wrong" in client.get(f"/api/videos/{vid2}").json()["error"]


def test_video_without_speech_is_ready_but_flagged(client):
    vid = upload(client, b"SILENT...", "music.mp4").json()["id"]
    client.jobs.wait_idle()
    v = client.get(f"/api/videos/{vid}").json()
    assert v["status"] == "ready" and v["has_speech"] is False


def test_rejects_bad_uploads(client):
    assert upload(client, b"x", "notes.pdf").status_code == 400
    assert upload(client, b"", "empty.mp4").status_code == 400


def test_video_file_supports_seeking(client):
    vid = upload(client, b"0123456789" * 100).json()["id"]
    client.jobs.wait_idle()
    r = client.get(f"/api/videos/{vid}/file", headers={"Range": "bytes=10-19"})
    assert r.status_code == 206 and r.content == b"0123456789"


def test_search_before_ready_and_unknown_ids(client):
    assert client.get("/api/videos/0123456789abcdef").status_code == 404
    assert client.get("/api/videos/../../etc/passwd").status_code == 404
    assert client.get("/api/videos/0123456789abcdef/search", params={"q": "x"}).status_code == 409


def test_delete_removes_video_and_index(client):
    vid = upload(client).json()["id"]
    client.jobs.wait_idle()
    assert client.delete(f"/api/videos/{vid}").status_code == 204
    assert client.get(f"/api/videos/{vid}").status_code == 404
    assert client.get("/api/videos").json() == []
    assert not list(Path(api.VIDEO_DIR).glob("*.mp4"))


# ---------------------------------------------------------------- visual search (opt-in)
def test_visual_is_off_by_default(client):
    vid = upload(client).json()["id"]
    client.jobs.wait_idle()
    assert client.get(f"/api/videos/{vid}").json()["visual"] == "off"
    res = client.get(f"/api/videos/{vid}/search", params={"q": "gradient descent"}).json()
    assert res["visual_enabled"] is False and res["visual"] == []
    assert PIPELINE_CALLS == [{"video_id": vid, "audio": True, "visual": False}]


def test_upload_with_visual_finds_slide_text_and_pictures(client):
    vid = upload(client, visual=True).json()["id"]
    client.jobs.wait_idle()
    assert client.get(f"/api/videos/{vid}").json()["visual"] == "on"

    res = client.get(f"/api/videos/{vid}/search", params={"q": "gradient descent"}).json()
    assert res["visual_enabled"] is True
    slide = res["visual"][0]
    assert slide["source"] == "text" and slide["frame"] == 1 and "Gradient Descent" in slide["text"]
    assert slide["jump_time"] == pytest.approx(20.0 - store.FRAME_LEAD_IN)

    res = client.get(f"/api/videos/{vid}/search", params={"q": "a photo of a cat"}).json()
    pics = [r for r in res["visual"] if r["source"] == "image"]
    assert pics and pics[0]["frame"] == CAT_FRAME


def test_unrelated_query_shows_nothing_on_screen(client):
    vid = upload(client, visual=True).json()["id"]
    client.jobs.wait_idle()
    res = client.get(f"/api/videos/{vid}/search", params={"q": "zebra xylophone"}).json()
    assert res["visual"] == []


def test_enable_visual_later_without_retranscribing(client):
    vid = upload(client).json()["id"]
    client.jobs.wait_idle()
    r = client.post(f"/api/videos/{vid}/visual")
    assert r.status_code == 202 and r.json()["visual"] in ("processing", "on")
    client.jobs.wait_idle()
    assert client.get(f"/api/videos/{vid}").json()["visual"] == "on"
    assert PIPELINE_CALLS[-1] == {"video_id": vid, "audio": False, "visual": True}


def test_visual_failure_keeps_speech_search_working(client):
    vid = upload(client, b"VISFAIL...", "lecture.mp4", visual=True).json()["id"]
    client.jobs.wait_idle()
    v = client.get(f"/api/videos/{vid}").json()
    assert v["status"] == "ready" and v["visual"] == "error" and "frames" in v["visual_error"]
    assert client.get(f"/api/videos/{vid}/search", params={"q": "learning rate"}).json()["results"]


def test_frame_thumbnails_and_cleanup(client):
    vid = upload(client, visual=True).json()["id"]
    client.jobs.wait_idle()
    assert client.get(f"/api/videos/{vid}/frames/1.jpg").status_code == 200
    assert client.get(f"/api/videos/{vid}/frames/99.jpg").status_code == 404
    client.delete(f"/api/videos/{vid}")
    assert not Path(api.FRAMES_DIR, vid).exists()
    assert not store.has_visual(vid)

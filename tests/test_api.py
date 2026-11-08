from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import backend.api as api
import backend.store as store
from backend.jobs import JobManager
from backend.processor import ProcessingError
from test_store import LECTURE, _fake_embed, _fake_rerank


def fake_pipeline(video_id, path, name, report):
    """Stands in for ffmpeg + Whisper: indexes a canned transcript."""
    report("Transcribing speech", 0.5)
    with open(path, "rb") as f:
        content = f.read()
    if content.startswith(b"NOAUDIO"):
        raise ProcessingError("This video has no audio track, so there is no speech to search.")
    if content.startswith(b"CRASH"):
        raise RuntimeError("boom")
    sentences = [] if content.startswith(b"SILENT") else LECTURE
    store.index_video(video_id, name, path, 60.0, sentences)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(store, "DB_PATH", str(tmp_path / "test.sqlite3"))
    monkeypatch.setattr(store, "_embed", _fake_embed)
    monkeypatch.setattr(store, "_rerank", _fake_rerank)
    monkeypatch.setattr(api, "VIDEO_DIR", str(tmp_path / "videos"))
    store._emb_cache.clear()
    jobs = JobManager(pipeline=fake_pipeline)
    with TestClient(api.create_app(jobs)) as c:
        c.jobs = jobs
        yield c


def upload(client, content=b"VIDEO-BYTES-1", filename="Lecture 1.mp4"):
    return client.post("/api/videos", params={"filename": filename}, content=content)


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

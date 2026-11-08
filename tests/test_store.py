import re

import numpy as np
import pytest

import backend.store as store
from backend.store import build_chunks, fts_query, rrf_fuse

VOCAB_DIM = 512


def _fake_embed(texts):
    """Hashed bag-of-words vectors: similar wording -> similar vectors."""
    out = np.zeros((len(texts), VOCAB_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"\w+", t.lower()):
            out[i, hash(w) % VOCAB_DIM] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)


def _fake_rerank(query, texts):
    """Score = number of query words present (a crude stand-in for the cross-encoder)."""
    q = set(re.findall(r"\w+", query.lower()))
    return [float(len(q & set(re.findall(r"\w+", t.lower())))) * 2 - 1 for t in texts]


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(store, "DB_PATH", str(tmp_path / "test.sqlite3"))
    monkeypatch.setattr(store, "_embed", _fake_embed)
    monkeypatch.setattr(store, "_rerank", _fake_rerank)
    store._emb_cache.clear()
    yield


def _sentences(texts, seconds_each=6.0):
    return [{"start": i * seconds_each, "end": (i + 1) * seconds_each - 0.5, "text": t}
            for i, t in enumerate(texts)]


LECTURE = _sentences([
    "Welcome to today's lecture on optimisation.",
    "First we review linear algebra basics.",
    "Matrices can be multiplied when shapes agree.",
    "Now let us talk about gradient descent.",
    "Gradient descent takes small steps downhill along the gradient.",
    "The learning rate controls the step size.",
    "Next we cover Dijkstra's shortest path algorithm.",
    "It uses a priority queue of unvisited nodes.",
    "Finally the exam will be on the fourteenth of December.",
    "Thanks for listening.",
])


# ---------------------------------------------------------------- chunking
def test_chunks_never_cut_sentences_and_overlap_by_one():
    chunks = build_chunks(LECTURE, target=12, max_seconds=20)
    assert chunks[0]["first_sent"] == 0
    assert chunks[-1]["last_sent"] == len(LECTURE) - 1
    for a, b in zip(chunks, chunks[1:]):
        assert b["first_sent"] == a["last_sent"]          # one shared sentence
    for c in chunks:
        assert c["end"] - c["start"] <= 20
        assert c["start"] == LECTURE[c["first_sent"]]["start"]


def test_single_long_sentence_still_progresses():
    sents = [{"start": 0, "end": 40, "text": "a"}, {"start": 40, "end": 80, "text": "b"}]
    chunks = build_chunks(sents, target=20, max_seconds=30)
    assert [(c["first_sent"], c["last_sent"]) for c in chunks] == [(0, 0), (1, 1)]


def test_build_chunks_empty():
    assert build_chunks([]) == []


# ---------------------------------------------------------------- helpers
def test_fts_query_is_safe_against_fts_syntax():
    assert fts_query('what is "AND" OR (x)*?') == '"what" OR "is" OR "and" OR "or" OR "x"'
    assert fts_query("???") == ""


def test_rrf_rewards_agreement():
    assert rrf_fuse([1, 2, 3], [3, 1, 4])[0] == 1
    assert set(rrf_fuse([1], [2])) == {1, 2}


# ---------------------------------------------------------------- library
def test_index_and_search_pinpoints_the_sentence():
    n = store.index_video("v1", "lec.mp4", "/x/lec.mp4", 60.0, LECTURE)
    assert n > 0 and store.is_indexed("v1")

    results = store.search("v1", "what controls the learning rate step size")
    top = results[0]
    assert top["sentence"] == "The learning rate controls the step size."
    assert top["jump_time"] == pytest.approx(LECTURE[5]["start"] - store.LEAD_IN_SECONDS)
    assert top["start"] <= LECTURE[5]["start"] <= top["end"]


def test_results_are_sorted_and_deduplicated():
    store.index_video("v1", "lec.mp4", "/x", 60.0, LECTURE)
    results = store.search("v1", "gradient descent", top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    jumps = [r["jump_time"] for r in results]
    assert all(abs(a - b) >= store.DEDUPE_SECONDS for i, a in enumerate(jumps) for b in jumps[i + 1:])


def test_exact_keyword_is_found():
    store.index_video("v1", "lec.mp4", "/x", 60.0, LECTURE)
    assert "Dijkstra" in store.search("v1", "dijkstra")[0]["sentence"]


def test_unrelated_query_returns_nothing():
    store.index_video("v1", "lec.mp4", "/x", 60.0, LECTURE)
    store.MIN_SCORE, old = 0.0, store.MIN_SCORE   # fake reranker scores -1 for no overlap
    try:
        assert store.search("v1", "zebra xylophone") == []
    finally:
        store.MIN_SCORE = old


def test_no_speech_video_is_recorded_so_it_is_not_reprocessed():
    assert store.index_video("silent", "s.mp4", "/x", 30.0, []) == 0
    assert store.is_indexed("silent")
    assert store.get_video("silent")["chunks"] == 0
    assert store.search("silent", "anything") == []


def test_reindex_replaces_instead_of_duplicating():
    store.index_video("v1", "lec.mp4", "/x", 60.0, LECTURE)
    first = store.get_video("v1")["chunks"]
    store.index_video("v1", "lec.mp4", "/x", 60.0, LECTURE)
    assert store.get_video("v1")["chunks"] == first


def test_failed_index_leaves_no_partial_video():
    bad = LECTURE[:3] + [{"start": 99, "end": 100, "text": None}]  # violates NOT NULL
    with pytest.raises(Exception):
        store.index_video("v2", "bad.mp4", "/x", 60.0, bad)
    assert not store.is_indexed("v2")


def test_search_is_scoped_to_one_video():
    store.index_video("v1", "a.mp4", "/x", 60.0, LECTURE)
    store.index_video("v2", "b.mp4", "/x", 10.0, _sentences(["Photosynthesis happens in leaves."]))
    assert all("Photosynthesis" not in r["text"] for r in store.search("v1", "photosynthesis leaves"))

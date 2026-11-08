"""
store.py
SQLite-backed lecture index and search.

Indexing:  sentences -> overlapping sentence-aligned chunks -> embeddings,
           stored in one SQLite file (plus an FTS5 keyword index).
Search:    keyword (BM25) + meaning (embeddings) candidates, fused with
           reciprocal-rank fusion, reranked by a cross-encoder, then each hit is
           pinpointed to the single sentence that best answers the query.
"""

import os
import re
import sqlite3
import threading
import time
import logging
from contextlib import contextmanager

import numpy as np

from backend.paths import DATA_DIR, DB_PATH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBED_MODEL_NAME = "all-mpnet-base-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"

CHUNK_TARGET_SECONDS = 20.0   # grow a chunk until it covers about this much speech
CHUNK_MAX_SECONDS = 30.0      # ...but never beyond this
CANDIDATES = 25               # chunks passed from retrieval to the reranker
RRF_K = 60                    # reciprocal-rank-fusion constant
STRONG_SCORE = 1.5            # reranker logit at/above which a match is "strong"
MIN_SCORE = -6.0              # below this a chunk is unrelated and is dropped
LEAD_IN_SECONDS = 1.0         # jump slightly before the sentence starts
DEDUPE_SECONDS = 20.0         # results whose jump points are closer are merged

_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    path       TEXT NOT NULL,
    duration   REAL NOT NULL,
    indexed_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sentences (
    video_id TEXT    NOT NULL,
    idx      INTEGER NOT NULL,
    start    REAL    NOT NULL,
    end      REAL    NOT NULL,
    text     TEXT    NOT NULL,
    PRIMARY KEY (video_id, idx)
);
CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY,
    video_id   TEXT    NOT NULL,
    first_sent INTEGER NOT NULL,
    last_sent  INTEGER NOT NULL,
    start      REAL    NOT NULL,
    end        REAL    NOT NULL,
    text       TEXT    NOT NULL,
    embedding  BLOB    NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_video ON chunks(video_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, video_id UNINDEXED);
"""

# ---------------------------------------------------------------------------
# Models (loaded once per process; Streamlit runs sessions on threads)
# ---------------------------------------------------------------------------
_model_lock = threading.Lock()
_embed_model = None
_reranker = None


def _get_embed_model():
    global _embed_model
    with _model_lock:
        if _embed_model is None:
            from sentence_transformers import SentenceTransformer
            _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def _get_reranker():
    global _reranker
    with _model_lock:
        if _reranker is None:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


def _embed(texts: list[str]) -> np.ndarray:
    vecs = _get_embed_model().encode(texts, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


def _rerank(query: str, texts: list[str]) -> list[float]:
    if not texts:
        return []
    return [float(s) for s in _get_reranker().predict([[query, t] for t in texts])]


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
@contextmanager
def _connect():
    """Yield a connection inside one transaction, then close it.

    A fresh connection per call keeps this safe across Streamlit's threads.
    The transaction commits on success and rolls back on any exception.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        with conn:
            yield conn
    finally:
        conn.close()


# Per-video embedding matrices, so a search does not re-read every blob.
_emb_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_emb_cache_lock = threading.Lock()


def _video_embeddings(conn: sqlite3.Connection, video_id: str) -> tuple[np.ndarray, np.ndarray]:
    with _emb_cache_lock:
        if video_id not in _emb_cache:
            rows = conn.execute(
                "SELECT id, embedding FROM chunks WHERE video_id = ? ORDER BY id",
                (video_id,),
            ).fetchall()
            ids = np.array([r["id"] for r in rows], dtype=np.int64)
            mat = (np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
                   if rows else np.zeros((0, 0), dtype=np.float32))
            _emb_cache[video_id] = (ids, mat)
        return _emb_cache[video_id]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def build_chunks(
    sentences: list[dict],
    target: float = CHUNK_TARGET_SECONDS,
    max_seconds: float = CHUNK_MAX_SECONDS,
) -> list[dict]:
    """Group sentences into overlapping chunks that never cut a sentence.

    Each chunk grows sentence by sentence until it covers *target* seconds
    (without exceeding *max_seconds*); consecutive chunks share one sentence
    so an idea spanning a boundary is still found.

    Returns:
        ``[{first_sent, last_sent, start, end, text}]`` (sentence indices
        are inclusive positions in *sentences*).
    """
    chunks: list[dict] = []
    n = len(sentences)
    i = 0
    while i < n:
        j = i
        while (j + 1 < n
               and sentences[j]["end"] - sentences[i]["start"] < target
               and sentences[j + 1]["end"] - sentences[i]["start"] <= max_seconds):
            j += 1
        chunks.append({
            "first_sent": i,
            "last_sent": j,
            "start": sentences[i]["start"],
            "end": sentences[j]["end"],
            "text": " ".join(s["text"] for s in sentences[i:j + 1]),
        })
        if j + 1 >= n:
            break
        i = j if j > i else j + 1   # overlap by one sentence
    return chunks


# ---------------------------------------------------------------------------
# Library API
# ---------------------------------------------------------------------------
def is_indexed(video_id: str) -> bool:
    with _connect() as conn:
        return conn.execute("SELECT 1 FROM videos WHERE id = ?", (video_id,)).fetchone() is not None


def get_video(video_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
        if row is None:
            return None
        video = dict(row)
        video["chunks"] = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE video_id = ?", (video_id,)
        ).fetchone()[0]
        return video


def delete_video(video_id: str) -> None:
    with _connect() as conn:
        _delete_rows(conn, video_id)
    with _emb_cache_lock:
        _emb_cache.pop(video_id, None)


def _delete_rows(conn: sqlite3.Connection, video_id: str) -> None:
    conn.execute("DELETE FROM chunks_fts WHERE video_id = ?", (video_id,))
    conn.execute("DELETE FROM chunks WHERE video_id = ?", (video_id,))
    conn.execute("DELETE FROM sentences WHERE video_id = ?", (video_id,))
    conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))


def index_video(
    video_id: str,
    name: str,
    path: str,
    duration: float,
    sentences: list[dict],
) -> int:
    """Store a transcribed video. All-or-nothing: a crash leaves no partial index.

    A video with no speech is still recorded (with zero chunks) so it is not
    re-processed on every upload.

    Returns:
        Number of chunks indexed.
    """
    chunks = build_chunks(sentences)
    embeddings = _embed([c["text"] for c in chunks]) if chunks else None

    with _connect() as conn:  # one transaction: all rows land, or none do
        _delete_rows(conn, video_id)
        conn.execute(
            "INSERT INTO videos (id, name, path, duration, indexed_at) VALUES (?, ?, ?, ?, ?)",
            (video_id, name, path, duration, time.time()),
        )
        conn.executemany(
            "INSERT INTO sentences (video_id, idx, start, end, text) VALUES (?, ?, ?, ?, ?)",
            [(video_id, i, s["start"], s["end"], s["text"]) for i, s in enumerate(sentences)],
        )
        for c, emb in zip(chunks, embeddings if embeddings is not None else []):
            cur = conn.execute(
                "INSERT INTO chunks (video_id, first_sent, last_sent, start, end, text, embedding)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (video_id, c["first_sent"], c["last_sent"], c["start"], c["end"],
                 c["text"], emb.tobytes()),
            )
            conn.execute(
                "INSERT INTO chunks_fts (rowid, text, video_id) VALUES (?, ?, ?)",
                (cur.lastrowid, c["text"], video_id),
            )
    with _emb_cache_lock:
        _emb_cache.pop(video_id, None)
    return len(chunks)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 query: any word may match (OR)."""
    words = re.findall(r"\w+", query.lower())
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(words))


def rrf_fuse(*rankings: list[int], k: int = RRF_K) -> list[int]:
    """Reciprocal-rank fusion of several ranked id lists (best first)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for pos, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + pos + 1)
    return sorted(scores, key=scores.get, reverse=True)


def _keyword_candidates(conn, video_id: str, query: str, limit: int) -> list[int]:
    q = fts_query(query)
    if not q:
        return []
    rows = conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? AND video_id = ?"
        " ORDER BY bm25(chunks_fts) LIMIT ?",
        (q, video_id, limit),
    ).fetchall()
    return [r[0] for r in rows]


def _meaning_candidates(conn, video_id: str, query: str, limit: int) -> list[int]:
    ids, mat = _video_embeddings(conn, video_id)
    if len(ids) == 0:
        return []
    sims = mat @ _embed([query])[0]
    top = np.argsort(-sims)[:limit]
    return [int(ids[i]) for i in top]


def search(video_id: str, query: str, top_k: int = 5) -> list[dict]:
    """Find the moments in a video that answer *query*.

    Returns:
        Up to *top_k* results, best first::

            {jump_time, sentence, start, end, text, score, strong}

        ``jump_time`` is where playback should start (just before the best
        sentence); ``start``/``end``/``text`` describe the surrounding chunk.
        ``strong`` is False for weak matches, which are still shown so a
        correct answer is never hidden by a hard cutoff.
    """
    query = query.strip()
    if not query:
        return []

    with _connect() as conn:
        candidates = rrf_fuse(
            _keyword_candidates(conn, video_id, query, CANDIDATES),
            _meaning_candidates(conn, video_id, query, CANDIDATES),
        )[:CANDIDATES]
        if not candidates:
            return []

        marks = ",".join("?" * len(candidates))
        rows = {r["id"]: dict(r) for r in conn.execute(
            f"SELECT id, first_sent, last_sent, start, end, text FROM chunks WHERE id IN ({marks})",
            candidates,
        )}
        chunks = [rows[c] for c in candidates if c in rows]
        for c, score in zip(chunks, _rerank(query, [c["text"] for c in chunks])):
            c["score"] = score

        # Keep a few spares: neighbouring chunks overlap and collapse in dedupe.
        chunks = sorted((c for c in chunks if c["score"] >= MIN_SCORE),
                        key=lambda c: c["score"], reverse=True)[:top_k * 2]
        if not chunks:
            return []

        # Pinpoint: rerank the sentences inside each chunk in one batch.
        sents_by_chunk = []
        for c in chunks:
            sents_by_chunk.append([dict(r) for r in conn.execute(
                "SELECT start, text FROM sentences WHERE video_id = ? AND idx BETWEEN ? AND ?"
                " ORDER BY idx",
                (video_id, c["first_sent"], c["last_sent"]),
            )])
    flat = [s for sents in sents_by_chunk for s in sents]
    for s, score in zip(flat, _rerank(query, [s["text"] for s in flat])):
        s["score"] = score

    results: list[dict] = []
    for c, sents in zip(chunks, sents_by_chunk):
        best = max(sents, key=lambda s: s["score"]) if sents else {"start": c["start"], "text": c["text"]}
        jump = round(max(0.0, best["start"] - LEAD_IN_SECONDS), 2)
        if any(abs(jump - r["jump_time"]) < DEDUPE_SECONDS for r in results):
            continue  # same moment already reported by a higher-ranked chunk
        results.append({
            "jump_time": jump,
            "sentence": best["text"],
            "start": c["start"],
            "end": c["end"],
            "text": c["text"],
            "score": round(c["score"], 3),
            "strong": c["score"] >= STRONG_SCORE,
        })
        if len(results) == top_k:
            break
    return results

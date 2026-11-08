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
RELEVANT_SCORE = 0.0          # reranker logit at/above which a passage is "about" the query
EXPAND_ROUNDS = 3             # rounds of scoring neighbours to find a section's full extent
SECTION_BONUS = 1.0           # topic searches: rank boost per extra passage in a section (max 3)
TOPIC_MAX_WORDS = 3           # short non-question searches are treated as topics
TOPIC_START_MARGIN = 2.5      # topic jump = first sentence scoring within this of the best
EXPLAINED_MIN_SECONDS = 60.0  # a section this long is an explanation, shorter is a mention

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

-- Optional visual index (only for videos where the user enabled it).
CREATE TABLE IF NOT EXISTS visual_index (
    video_id   TEXT PRIMARY KEY,
    frames     INTEGER NOT NULL,
    indexed_at REAL    NOT NULL
);
CREATE TABLE IF NOT EXISTS frames (
    id             INTEGER PRIMARY KEY,
    video_id       TEXT    NOT NULL,
    idx            INTEGER NOT NULL,
    start          REAL    NOT NULL,
    end            REAL    NOT NULL,
    text           TEXT    NOT NULL,
    text_embedding BLOB,
    clip_embedding BLOB    NOT NULL
);
CREATE INDEX IF NOT EXISTS frames_video ON frames(video_id);
CREATE VIRTUAL TABLE IF NOT EXISTS frames_fts USING fts5(text, video_id UNINDEXED);
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
        video["visual"] = conn.execute(
            "SELECT 1 FROM visual_index WHERE video_id = ?", (video_id,)
        ).fetchone() is not None
        return video


def list_videos() -> list[dict]:
    """All indexed videos, newest first, with their passage counts."""
    with _connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT v.id, v.name, v.path, v.duration, v.indexed_at,"
            "       (SELECT COUNT(*) FROM chunks c WHERE c.video_id = v.id) AS chunks,"
            "       EXISTS (SELECT 1 FROM visual_index x WHERE x.video_id = v.id) AS visual"
            " FROM videos v ORDER BY v.indexed_at DESC"
        )]


def delete_video(video_id: str) -> None:
    with _connect() as conn:
        _delete_rows(conn, video_id)
        _delete_frames(conn, video_id)
    with _emb_cache_lock:
        _emb_cache.pop(video_id, None)
        _frame_cache.pop(video_id, None)


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


_QUESTION_WORDS = {
    "what", "why", "how", "when", "who", "whom", "whose", "where", "which",
    "is", "are", "was", "were", "do", "does", "did", "can", "could",
    "should", "would", "will",
}
_TOPIC_PREFIXES = ("explain", "define", "describe", "overview of", "introduction to",
                   "intro to", "about")


def is_topic_query(query: str) -> bool:
    """True for "show me where X is covered" searches, False for specific questions.

    Topic searches ("brown sugar", "explain gradient descent") should land where
    the lecturer *starts* explaining the topic; questions ("who invented sugar
    cubes?") should land on the sentence that answers them.
    """
    q = query.lower().strip()
    if q.startswith(_TOPIC_PREFIXES):
        return True
    words = re.findall(r"[a-z0-9']+", q)
    if not words or "?" in q or words[0] in _QUESTION_WORDS:
        return False
    return len(words) <= TOPIC_MAX_WORDS


def group_sections(chunks: list[dict], max_gap: int = 1) -> list[list[dict]]:
    """Group relevant chunks into sections of consecutive passages (by time).

    Chunk ids are consecutive within a video. Up to *max_gap* unlisted chunks
    may sit between two members of a section: lecturers often spend a passage
    on a detail without naming the topic ("it only works in early spring"),
    and that should not split the explanation in two.
    """
    sections: list[list[dict]] = []
    for c in sorted(chunks, key=lambda c: c["id"]):
        if sections and c["id"] - sections[-1][-1]["id"] <= 1 + max_gap:
            sections[-1].append(c)
        else:
            sections.append([c])
    return sections


_CHUNK_COLS = "id, first_sent, last_sent, start, end, text"


def _score_chunks(conn, video_id: str, query: str) -> dict[int, dict]:
    """Retrieve candidate chunks, rerank them, and grow relevant runs outward.

    Neighbours of relevant chunks are scored too (a few rounds), so a section's
    true extent is known even if its middle or edges were not retrieved.
    """
    candidates = rrf_fuse(
        _keyword_candidates(conn, video_id, query, CANDIDATES),
        _meaning_candidates(conn, video_id, query, CANDIDATES),
    )[:CANDIDATES]
    scored: dict[int, dict] = {}
    to_score = candidates
    for _ in range(1 + EXPAND_ROUNDS):
        if not to_score:
            break
        marks = ",".join("?" * len(to_score))
        rows = [dict(r) for r in conn.execute(
            f"SELECT {_CHUNK_COLS} FROM chunks WHERE video_id = ? AND id IN ({marks})",
            (video_id, *to_score),
        )]
        for c, score in zip(rows, _rerank(query, [c["text"] for c in rows])):
            c["score"] = score
            scored[c["id"]] = c
        # Next round: unscored neighbours (adjacent chunk ids) of relevant chunks.
        to_score = sorted({
            n for c in rows if c["score"] >= RELEVANT_SCORE
            for n in (c["id"] - 1, c["id"] + 1) if n not in scored
        })
    return scored


def search(video_id: str, query: str, top_k: int = 5) -> list[dict]:
    """Find the moments in a video that answer *query*.

    Relevant passages that follow one another are grouped into a *section*.
    A section of several passages is where a topic is explained; a lone
    passage is a passing mention. Explanations rank above mentions for topic
    searches, and each result appears once instead of once per passage.

    Returns:
        Up to *top_k* results, best first::

            {jump_time, sentence, start, end, text, score, strong, kind}

        ``jump_time`` is where playback should start: the start of the
        explanation for topic searches, or the answering sentence for
        questions. ``start``/``end`` span the whole section, ``text`` is the
        passage around ``sentence``. ``kind`` is ``"explained"`` (a section of at
        least a minute) or ``"mention"``. ``strong`` is False for weak matches, which
        are still shown so a correct answer is never hidden by a hard cutoff.
    """
    query = query.strip()
    if not query:
        return []
    topic = is_topic_query(query)

    with _connect() as conn:
        scored = _score_chunks(conn, video_id, query)
        relevant = [c for c in scored.values() if c["score"] >= RELEVANT_SCORE]
        sections = group_sections(relevant)
        # Weakly related lone passages fill any remaining slots, so a correct
        # but low-confidence answer is still offered.
        covered = {i for sec in sections for i in range(sec[0]["id"], sec[-1]["id"] + 1)}
        sections += [[c] for c in scored.values()
                     if c["id"] not in covered and c["score"] >= MIN_SCORE]

        def explained(sec: list[dict]) -> bool:
            return sec[-1]["end"] - sec[0]["start"] >= EXPLAINED_MIN_SECONDS

        def rank(sec: list[dict]) -> float:
            best = max(c["score"] for c in sec)
            if topic and explained(sec) and best >= RELEVANT_SCORE:
                best += SECTION_BONUS * min(len(sec) - 1, 3)
            return best

        sections = sorted(sections, key=rank, reverse=True)[:top_k * 2]
        if not sections:
            return []

        # Sentences to choose the jump point from: for a topic, the opening of
        # the section; for a question, the section's single best passage.
        sents_by_sec = []
        for sec in sections:
            pool = sec[:2] if topic else [max(sec, key=lambda c: c["score"])]
            lo, hi = min(c["first_sent"] for c in pool), max(c["last_sent"] for c in pool)
            sents_by_sec.append([dict(r) for r in conn.execute(
                "SELECT idx, start, text FROM sentences WHERE video_id = ? AND idx BETWEEN ? AND ?"
                " ORDER BY idx", (video_id, lo, hi),
            )])

    flat = [s for sents in sents_by_sec for s in sents]
    for s, score in zip(flat, _rerank(query, [s["text"] for s in flat])):
        s["score"] = score

    results: list[dict] = []
    for sec, sents in zip(sections, sents_by_sec):
        if not sents:
            continue
        best = max(sents, key=lambda s: s["score"])
        if topic and explained(sec):
            # Where the explanation starts: the first sentence that is itself
            # about the topic and nearly as relevant as the best one.
            floor = max(best["score"] - TOPIC_START_MARGIN, RELEVANT_SCORE)
            best = next((s for s in sents if s["score"] >= floor), best)
        jump = round(max(0.0, best["start"] - LEAD_IN_SECONDS), 2)
        if any(abs(jump - r["jump_time"]) < DEDUPE_SECONDS for r in results):
            continue  # same moment already reported by a higher-ranked section
        host = next((c for c in sec if c["first_sent"] <= best["idx"] <= c["last_sent"]), sec[0])
        top = max(c["score"] for c in sec)
        results.append({
            "jump_time": jump,
            "sentence": best["text"],
            "start": sec[0]["start"],
            "end": sec[-1]["end"],
            "text": host["text"],
            "score": round(top, 3),
            "strong": top >= STRONG_SCORE,
            "kind": "explained" if explained(sec) else "mention",
        })
        if len(results) == top_k:
            break
    return results


# ---------------------------------------------------------------------------
# Visual index (optional): key frames with on-screen text and CLIP embeddings
# ---------------------------------------------------------------------------
VISUAL_CANDIDATES = 15        # OCR frames passed to the reranker
CLIP_MIN_SIMILARITY = 0.25    # CLIP cosine below this is never a match
CLIP_MIN_Z = 2.5              # ...and the frame must stand out from the rest of the video
CLIP_STRONG_Z = 4.5           # CLIP is overconfident on frames with look-alike words, so "strong" is rare
FRAME_LEAD_IN = 0.5

_frame_cache: dict[str, dict] = {}


def _delete_frames(conn: sqlite3.Connection, video_id: str) -> None:
    conn.execute("DELETE FROM frames_fts WHERE video_id = ?", (video_id,))
    conn.execute("DELETE FROM frames WHERE video_id = ?", (video_id,))
    conn.execute("DELETE FROM visual_index WHERE video_id = ?", (video_id,))


def has_visual(video_id: str) -> bool:
    with _connect() as conn:
        return conn.execute("SELECT 1 FROM visual_index WHERE video_id = ?",
                            (video_id,)).fetchone() is not None


def index_frames(video_id: str, frames: list[dict]) -> int:
    """Store key frames ``[{start, end, text, embedding}]``. All-or-nothing.

    A frame's ``idx`` is its position, which is also its thumbnail's name.
    """
    texts = [f["text"] for f in frames]
    with_text = [i for i, t in enumerate(texts) if t]
    text_vecs = dict(zip(with_text, _embed([texts[i] for i in with_text]))) if with_text else {}

    with _connect() as conn:
        _delete_frames(conn, video_id)
        for i, f in enumerate(frames):
            tv = text_vecs.get(i)
            cur = conn.execute(
                "INSERT INTO frames (video_id, idx, start, end, text, text_embedding, clip_embedding)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (video_id, i, f["start"], f["end"], f["text"],
                 tv.tobytes() if tv is not None else None,
                 np.asarray(f["embedding"], dtype=np.float32).tobytes()),
            )
            if f["text"]:
                conn.execute("INSERT INTO frames_fts (rowid, text, video_id) VALUES (?, ?, ?)",
                             (cur.lastrowid, f["text"], video_id))
        conn.execute("INSERT INTO visual_index (video_id, frames, indexed_at) VALUES (?, ?, ?)",
                     (video_id, len(frames), time.time()))
    with _emb_cache_lock:
        _frame_cache.pop(video_id, None)
    return len(frames)


def _video_frames(conn: sqlite3.Connection, video_id: str) -> dict:
    with _emb_cache_lock:
        if video_id not in _frame_cache:
            rows = [dict(r) for r in conn.execute(
                "SELECT id, idx, start, end, text, text_embedding, clip_embedding"
                " FROM frames WHERE video_id = ? ORDER BY idx", (video_id,))]
            clip = (np.vstack([np.frombuffer(r["clip_embedding"], dtype=np.float32) for r in rows])
                    if rows else np.zeros((0, 0), dtype=np.float32))
            text_rows = [r for r in rows if r["text_embedding"] is not None]
            text_mat = (np.vstack([np.frombuffer(r["text_embedding"], dtype=np.float32) for r in text_rows])
                        if text_rows else np.zeros((0, 0), dtype=np.float32))
            for r in rows:
                del r["clip_embedding"], r["text_embedding"]
            _frame_cache[video_id] = {
                "rows": rows, "by_id": {r["id"]: r for r in rows}, "clip": clip,
                "text_ids": [r["id"] for r in text_rows], "text": text_mat,
            }
        return _frame_cache[video_id]


def _clip_text(query: str) -> np.ndarray:
    from backend import visual
    return visual.embed_text(query)


def search_visual(video_id: str, query: str, top_k: int = 4) -> list[dict]:
    """Moments where the query is *shown*: on-screen text, or a matching picture.

    Returns:
        Up to *top_k* results, best first::

            {jump_time, start, end, frame, source, text, score, strong}

        ``source`` is ``"text"`` (the words appear on screen, e.g. a slide)
        or ``"image"`` (the picture matches the description). ``frame`` is
        the key-frame number, used to show a thumbnail.
    """
    query = query.strip()
    if not query:
        return []
    with _connect() as conn:
        data = _video_frames(conn, video_id)
        if not data["rows"]:
            return []
        q = fts_query(query)
        keyword = [r[0] for r in conn.execute(
            "SELECT rowid FROM frames_fts WHERE frames_fts MATCH ? AND video_id = ?"
            " ORDER BY bm25(frames_fts) LIMIT ?", (q, video_id, VISUAL_CANDIDATES))] if q else []

    # On-screen text: keyword + meaning candidates, reranked like speech.
    meaning: list[int] = []
    if data["text_ids"]:
        sims = data["text"] @ _embed([query])[0]
        meaning = [data["text_ids"][i] for i in np.argsort(-sims)[:VISUAL_CANDIDATES]]
    cands = [data["by_id"][i] for i in rrf_fuse(keyword, meaning)[:VISUAL_CANDIDATES]
             if i in data["by_id"]]
    text_hits = [(f, "text", s, s >= STRONG_SCORE)
                 for f, s in zip(cands, _rerank(query, [f["text"] for f in cands]))
                 if s >= RELEVANT_SCORE]
    text_hits.sort(key=lambda h: h[2], reverse=True)

    # Pictures: CLIP similarity, kept only if the frame clearly stands out.
    image_hits = []
    if len(data["rows"]) >= 3:
        sims = data["clip"] @ _clip_text(query)
        mean, std = float(sims.mean()), float(sims.std()) or 1e-6
        for i in np.argsort(-sims)[:top_k * 3]:
            z = (float(sims[i]) - mean) / std
            if sims[i] >= CLIP_MIN_SIMILARITY and z >= CLIP_MIN_Z:
                image_hits.append((data["rows"][i], "image", z, z >= CLIP_STRONG_Z))

    # When something matches clearly, weak picture guesses are just noise.
    if any(h[3] for h in text_hits + image_hits):
        image_hits = [h for h in image_hits if h[3]]

    results: list[dict] = []
    for f, source, score, strong in text_hits + image_hits:
        jump = round(max(0.0, f["start"] - FRAME_LEAD_IN), 2)
        if any(abs(jump - r["jump_time"]) < DEDUPE_SECONDS for r in results):
            continue
        results.append({
            "jump_time": jump, "start": f["start"], "end": f["end"], "frame": f["idx"],
            "source": source, "text": f["text"], "score": round(float(score), 3),
            "strong": bool(strong),
        })
        if len(results) == top_k:
            break
    return results

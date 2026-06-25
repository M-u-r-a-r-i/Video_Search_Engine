"""
vector_store.py
Sliding-window chunking, CLIP visual embeddings, and ChromaDB-backed
semantic search with cross-encoder reranking and result merging.
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Python 3.14 compat: chromadb imports pydantic.v1 which cannot infer types
# for PEP 695 TypeAliasType objects.  Monkey-patch *before* importing chromadb.
# ---------------------------------------------------------------------------
if sys.version_info >= (3, 14):
    import pydantic.v1.fields as _pv1_fields
    _orig_set_default = _pv1_fields.ModelField._set_default_and_type

    def _patched_set_default(self: _pv1_fields.ModelField) -> None:  # type: ignore[override]
        try:
            _orig_set_default(self)
        except _pv1_fields.errors_.ConfigError:
            # Fall back to Any so chromadb.config.Settings can load
            from typing import Any
            self.outer_type_ = Any
            self.type_ = Any

    _pv1_fields.ModelField._set_default_and_type = _patched_set_default  # type: ignore[assignment]

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
EMBED_MODEL_NAME = "all-mpnet-base-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
WINDOW_SECONDS = 30
OVERLAP_SECONDS = 5
RELEVANCE_THRESHOLD = 0.0       # minimum cross-encoder score to keep
INTENT_CONFIDENCE = 1.5         # cross-encoder score above which = confident intent match
MERGE_GAP_TOLERANCE = 15        # seconds: merge chunks within this gap
TEXT_WEIGHT = 0.7               # weight for text-based search score
VISUAL_WEIGHT = 0.3             # weight for visual-based search score

# ---------------------------------------------------------------------------
# Singleton-ish helpers (created once per process)
# ---------------------------------------------------------------------------
_embed_model: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None
_chroma_client: chromadb.ClientAPI | None = None
_clip_model = None
_clip_processor = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


def _get_chroma_client() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _chroma_client


def _get_clip():
    """Lazily load CLIP model and processor."""
    global _clip_model, _clip_processor
    if _clip_model is None:
        try:
            from transformers import CLIPModel, CLIPProcessor
            _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
            _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        except ImportError:
            raise ImportError(
                "Install transformers and torch for visual search: "
                "pip install transformers torch Pillow"
            )
    return _clip_model, _clip_processor


# ---------------------------------------------------------------------------
# Sliding-window chunking
# ---------------------------------------------------------------------------

def _sliding_window_chunk(
    segments: list[dict],
    window: float = WINDOW_SECONDS,
    overlap: float = OVERLAP_SECONDS,
) -> list[dict]:
    """Merge Whisper segments into fixed-duration overlapping chunks.

    Args:
        segments: List of ``{start, end, text}`` dicts (sorted by time).
        window:   Window size in seconds (default 60).
        overlap:  Overlap between consecutive windows in seconds (default 10).

    Returns:
        List of chunk dicts with ``start_time``, ``end_time``, and ``text``.
    """
    if not segments:
        return []

    total_duration = segments[-1]["end"]
    step = window - overlap
    chunks: list[dict] = []

    win_start = 0.0
    while win_start < total_duration:
        win_end = win_start + window

        # Gather text from every segment that overlaps this window
        texts: list[str] = []
        for seg in segments:
            if seg["end"] <= win_start:
                continue
            if seg["start"] >= win_end:
                break
            texts.append(seg["text"])

        if texts:
            chunks.append(
                {
                    "start_time": round(win_start, 2),
                    "end_time": round(min(win_end, total_duration), 2),
                    "text": " ".join(texts),
                }
            )

        win_start += step

    return chunks


# ---------------------------------------------------------------------------
# Post-processing: merge adjacent relevant chunks
# ---------------------------------------------------------------------------

def _merge_adjacent_results(
    results: list[dict],
    gap_tolerance: float = MERGE_GAP_TOLERANCE,
) -> list[dict]:
    """Merge temporally adjacent or overlapping result chunks into spans.

    Chunks whose start/end times are within *gap_tolerance* seconds of each
    other are merged into a single consolidated result.  The highest score
    among merged chunks is kept, and texts are concatenated.

    Args:
        results:       Ranked result dicts ``{start_time, end_time, text, score}``.
        gap_tolerance: Maximum gap (seconds) to bridge when merging.

    Returns:
        A (typically shorter) list of merged result dicts.
    """
    if not results:
        return []

    # Sort by start time for merging
    ordered = sorted(results, key=lambda r: r["start_time"])

    merged: list[dict] = [ordered[0].copy()]
    for chunk in ordered[1:]:
        prev = merged[-1]
        # If this chunk starts within gap_tolerance of the previous chunk's end
        if chunk["start_time"] <= prev["end_time"] + gap_tolerance:
            prev["end_time"] = max(prev["end_time"], chunk["end_time"])
            prev["text"] = prev["text"] + " " + chunk["text"]
            prev["score"] = max(prev["score"], chunk["score"])
        else:
            merged.append(chunk.copy())

    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _safe_collection_name(video_id: str) -> str:
    """Sanitise a video ID into a valid ChromaDB collection name."""
    safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in video_id)[:63]
    if len(safe) < 3:
        safe = safe.ljust(3, "_")
    return safe


def has_video(video_id: str) -> bool:
    """Return True if this video already has a populated text collection."""
    client = _get_chroma_client()
    safe_name = _safe_collection_name(video_id)
    try:
        return client.get_collection(name=safe_name).count() > 0
    except Exception:
        return False


def delete_video(video_id: str) -> None:
    """Delete both the text and visual collections for a video (if present)."""
    client = _get_chroma_client()
    safe_name = _safe_collection_name(video_id)
    for name in (safe_name, safe_name + "_visual"):
        try:
            client.delete_collection(name=name)
        except Exception:
            logger.debug("No collection '%s' to delete", name, exc_info=True)


def add_video(video_id: str, segments: list[dict]) -> int:
    """Chunk, embed, and store segments for a video in ChromaDB.

    Args:
        video_id: Unique identifier for the video (used as collection name).
        segments: Raw Whisper segments ``[{start, end, text}, ...]``.

    Returns:
        Number of chunks stored.
    """
    chunks = _sliding_window_chunk(segments)
    if not chunks:
        return 0

    model = _get_embed_model()
    client = _get_chroma_client()
    safe_name = _safe_collection_name(video_id)

    collection = client.get_or_create_collection(name=safe_name, metadata={"hnsw:space": "cosine"})

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts).tolist()

    ids = [f"{safe_name}_{i}" for i in range(len(chunks))]
    metadatas = [
        {"start_time": c["start_time"], "end_time": c["end_time"]}
        for c in chunks
    ]

    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(chunks)


def add_video_frames(video_id: str, frames: list[dict]) -> int:
    """Embed video frames with CLIP and store in a visual ChromaDB collection.

    Args:
        video_id: Unique video identifier.
        frames:   List of ``{timestamp: float, frame_path: str}`` dicts.

    Returns:
        Number of frame embeddings stored.
    """
    if not frames:
        return 0

    from PIL import Image
    import torch

    clip_model, clip_processor = _get_clip()
    client = _get_chroma_client()

    safe_name = _safe_collection_name(video_id) + "_visual"

    # CLIP produces 512-dim embeddings; use cosine distance
    collection = client.get_or_create_collection(
        name=safe_name, metadata={"hnsw:space": "cosine"}
    )

    ids: list[str] = []
    embeddings: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for idx, frame in enumerate(frames):
        try:
            img = Image.open(frame["frame_path"]).convert("RGB")
            inputs = clip_processor(images=img, return_tensors="pt")
            with torch.no_grad():
                emb = clip_model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)  # L2 normalise
            embedding = emb.squeeze().tolist()

            ts = frame["timestamp"]
            ids.append(f"{safe_name}_{idx}")
            embeddings.append(embedding)
            documents.append(f"Frame at {ts:.1f}s")
            metadatas.append({"start_time": ts, "end_time": ts + 5.0})
        except Exception:
            continue

    if not ids:
        return 0

    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    return len(ids)


def search(video_id: str, query: str, top_k: int = 5) -> dict:
    """Perform combined text + visual semantic search with reranking and merging.

    Strategy:
      1. Over-fetch 5× text candidates with bi-encoder, rerank with cross-encoder.
      2. Query CLIP visual collection (if it exists) for frame-level matches.
      3. Fuse text and visual scores, apply relevance threshold.
      4. Merge adjacent/overlapping chunks into consolidated time spans.
      5. If no high-confidence results, fall back to keyword-level matches.

    Args:
        video_id: The video collection to search.
        query:    Natural-language query.
        top_k:    Number of final results to return.

    Returns:
        Dict with ``match_type`` ('exact' or 'keyword') and ``results`` list.
    """
    model = _get_embed_model()
    reranker = _get_reranker()
    client = _get_chroma_client()
    safe_name = _safe_collection_name(video_id)

    # ----- Text search with cross-encoder reranking -----
    text_candidates: list[dict] = []
    try:
        collection = client.get_collection(name=safe_name)
        fetch_k = min(top_k * 5, collection.count())
        if fetch_k > 0:
            query_embedding = model.encode([query]).tolist()
            raw = collection.query(query_embeddings=query_embedding, n_results=fetch_k)

            if raw and raw["documents"] and raw["documents"][0]:
                docs = raw["documents"][0]
                metas = raw["metadatas"][0]

                pairs = [[query, doc] for doc in docs]
                ce_scores = reranker.predict(pairs).tolist()

                for i, doc in enumerate(docs):
                    score = round(float(ce_scores[i]), 4)
                    if score < RELEVANCE_THRESHOLD:
                        continue
                    text_candidates.append({
                        "start_time": metas[i]["start_time"],
                        "end_time": metas[i]["end_time"],
                        "text": doc,
                        "score": score,
                        "raw_score": score,   # un-normalised cross-encoder score
                        "source": "text",
                    })
    except Exception:
        logger.warning("Text search failed for '%s'", safe_name, exc_info=True)

    # ----- Visual search with CLIP -----
    visual_candidates: list[dict] = []
    visual_name = safe_name + "_visual"
    try:
        vis_collection = client.get_collection(name=visual_name)
        if vis_collection.count() > 0:
            import torch
            clip_model, clip_processor = _get_clip()

            # Encode query text with CLIP's text encoder
            text_inputs = clip_processor(text=[query], return_tensors="pt", padding=True)
            with torch.no_grad():
                text_emb = clip_model.get_text_features(**text_inputs)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            query_clip_emb = text_emb.squeeze().tolist()

            vis_fetch = min(top_k * 3, vis_collection.count())
            vis_raw = vis_collection.query(
                query_embeddings=[query_clip_emb], n_results=vis_fetch
            )

            if vis_raw and vis_raw["documents"] and vis_raw["documents"][0]:
                vis_docs = vis_raw["documents"][0]
                vis_metas = vis_raw["metadatas"][0]
                vis_distances = vis_raw.get("distances", [[]])[0]

                for i, doc in enumerate(vis_docs):
                    # Convert cosine distance → similarity score
                    sim = 1.0 - vis_distances[i] if vis_distances else 0.5
                    visual_candidates.append({
                        "start_time": vis_metas[i]["start_time"],
                        "end_time": vis_metas[i]["end_time"],
                        "text": doc,
                        "score": round(float(sim), 4),
                        "source": "visual",
                    })
    except Exception:
        logger.debug("Visual search skipped/failed for '%s'", visual_name, exc_info=True)

    # ----- Confidence gate (uses the *raw* cross-encoder score) -----
    # The cross-encoder emits an unbounded logit; INTENT_CONFIDENCE is calibrated
    # against that raw value, so it must be checked before any normalisation.
    confident = any(c["raw_score"] >= INTENT_CONFIDENCE for c in text_candidates)

    # ----- Fuse text + visual scores per time span -----
    fused = _fuse_scores(text_candidates, visual_candidates)
    if not fused:
        return {"match_type": "none", "results": []}

    fused.sort(key=lambda c: c["score"], reverse=True)
    top_results = fused[:top_k]
    match_type = "exact" if confident else "keyword"

    # Clean up internal keys before returning
    for c in top_results:
        c.pop("source", None)
        c.pop("raw_score", None)

    # Merge adjacent / overlapping chunks into consolidated time spans
    merged = _merge_adjacent_results(top_results)
    return {"match_type": match_type, "results": merged}


def _normalise(items: list[dict]) -> None:
    """Scale each item's ``score`` into the [0, 1] range, in place.

    When every score is equal (e.g. a single result), they are all equally
    relevant, so they are set to 1.0 rather than collapsing to 0.0.
    """
    if not items:
        return
    scores = [c["score"] for c in items]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        for c in items:
            c["score"] = 1.0
        return
    span = hi - lo
    for c in items:
        c["score"] = (c["score"] - lo) / span


def _fuse_scores(
    text_candidates: list[dict],
    visual_candidates: list[dict],
) -> list[dict]:
    """Combine text and visual candidates into a single weighted ranking.

    Scores are normalised within each modality, then a text span's score is
    *boosted* by the best visual match that overlaps it in time — so a moment
    supported by both what is said and what is shown ranks highest. Visual
    matches with no overlapping text survive on their own (visual-only) weight.
    """
    _normalise(text_candidates)
    _normalise(visual_candidates)

    def _overlaps(a: dict, b: dict) -> bool:
        return a["start_time"] < b["end_time"] and b["start_time"] < a["end_time"]

    fused: list[dict] = []
    used_visual: set[int] = set()

    for t in text_candidates:
        best_v = 0.0
        for vi, v in enumerate(visual_candidates):
            if _overlaps(t, v):
                used_visual.add(vi)
                best_v = max(best_v, v["score"])
        t["score"] = round(t["score"] * TEXT_WEIGHT + best_v * VISUAL_WEIGHT, 4)
        fused.append(t)

    # Visual-only spans (no overlapping text) contribute at their visual weight.
    for vi, v in enumerate(visual_candidates):
        if vi not in used_visual:
            v["score"] = round(v["score"] * VISUAL_WEIGHT, 4)
            fused.append(v)

    return fused

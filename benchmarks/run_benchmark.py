"""
Measure VideoRAG search accuracy on a real video.

    python benchmarks/run_benchmark.py <video file> benchmarks/sugars_queries.json

The video is indexed into benchmarks/.cache (separate from your app library),
then every query is searched and scored:

  on the second  - the top result jumps to within 5 s of the answer sentence
  right passage  - the top result's passage contains the answer sentence
  in top 5       - any of the 5 results contains the answer sentence
  out-of-scope   - queries with no answer should produce no *strong* match

Topic queries (a bare subject) count "on the second" only when the jump lands
where the explanation of that subject starts, not at a passing mention.
"""

import hashlib
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("VIDEORAG_DATA_DIR", os.path.join(HERE, ".cache"))
sys.path.insert(0, os.path.dirname(HERE))

from backend import processor, store  # noqa: E402  (env var must be set first)

ON_THE_SECOND = 5.0


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower())


def _answer_start(sentences: list[dict], phrase: str) -> float | None:
    """Start time of the sentence in which *phrase* begins (phrase may span sentences)."""
    target = _norm(phrase)
    joined, owners = "", []
    for s in sentences:
        piece = _norm(s["text"]) + " "
        joined += piece
        owners += [s["start"]] * len(piece)
    pos = joined.find(target)
    return owners[pos] if pos >= 0 else None


def _index(video_path: str) -> tuple[str, float]:
    with open(video_path, "rb") as f:
        video_id = hashlib.md5(f.read()).hexdigest()[:16]
    if store.is_indexed(video_id):
        return video_id, 0.0
    info = processor.probe_video(video_path)
    t0 = time.perf_counter()
    audio = processor.extract_audio(video_path, video_id)
    sentences = processor.transcribe_audio(audio, info["duration"])
    store.index_video(video_id, os.path.basename(video_path), video_path, info["duration"], sentences)
    processor.cleanup_audio(video_id)
    took = time.perf_counter() - t0
    print(f"Indexed {info['duration']/60:.1f} min of video in {took/60:.1f} min "
          f"({info['duration']/took:.1f}x real time), {len(sentences)} sentences")
    return video_id, took


def main(video_path: str, queries_path: str) -> None:
    with open(queries_path, encoding="utf-8") as f:
        spec = json.load(f)
    video_id, _ = _index(video_path)
    with store._connect() as conn:
        sentences = [dict(r) for r in conn.execute(
            "SELECT start, end, text FROM sentences WHERE video_id = ? ORDER BY idx", (video_id,))]

    tally = {"normal": [0, 0, 0, 0], "exact": [0, 0, 0, 0], "topic": [0, 0, 0, 0]}   # second, passage, top5, total
    oos_ok = oos_total = 0
    latencies = []
    for item in spec["queries"]:
        t0 = time.perf_counter()
        results = store.search(video_id, item["q"], top_k=5)
        latencies.append(time.perf_counter() - t0)

        if item["answer"] is None:
            oos_total += 1
            ok = not any(r["strong"] for r in results)
            oos_ok += ok
            print(f"{'OK  ' if ok else 'FAIL'} out-of-scope ({len(results)} weak results)  {item['q']}")
            continue

        phrases = item["answer"] if isinstance(item["answer"], list) else [item["answer"]]
        starts = [a for a in (_answer_start(sentences, p) for p in phrases) if a is not None]
        row = tally[item.get("kind", "normal")]
        row[3] += 1
        if not starts:
            print(f"SKIP answer phrase not in transcript: {item['answer']!r}")
            continue
        ans = starts[0]
        contains = lambda r: any(r["start"] <= a <= r["end"] for a in starts)  # noqa: E731
        top = results[0] if results else None
        on_sec = bool(top) and any(
            abs(top["jump_time"] - (a - store.LEAD_IN_SECONDS)) <= ON_THE_SECOND for a in starts)
        passage = bool(top) and contains(top)
        top5 = any(contains(r) for r in results)
        row[0] += on_sec; row[1] += passage; row[2] += top5
        label = "EXACT" if on_sec else ("NEAR " if passage else ("TOP5 " if top5 else "MISS "))
        got = f"{top['jump_time']:7.1f}s {'strong' if top['strong'] else 'weak  '} {top.get('kind', ''):9s}" if top else "   none        "
        print(f"{label} answer at {ans:7.1f}s  jump {got}  {item['q']}")

    print()
    for kind, (sec, pas, t5, n) in tally.items():
        if n:
            print(f"{kind:7s} queries: on the second {sec}/{n} | right passage {pas}/{n} | in top 5 {t5}/{n}")
    print(f"out-of-scope: no strong match {oos_ok}/{oos_total}")
    warm = latencies[1:] or latencies
    print(f"search latency: {sum(warm)/len(warm):.2f}s avg (first query {latencies[0]:.1f}s incl. model load)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])

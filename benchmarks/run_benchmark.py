"""
Measure VideoRAG search accuracy on a real video.

    python benchmarks/run_benchmark.py <video file> benchmarks/sugars_queries.json
    python benchmarks/run_benchmark.py <video file> benchmarks/universe_queries.json --visual

The video is indexed into benchmarks/.cache (separate from your app library;
set VIDEORAG_DATA_DIR to reuse another library), then every query is searched
and scored:

  on the second  - the top speech result jumps to within 5 s of the answer sentence
  right passage  - the top speech result's section contains the answer sentence
  in top 5       - any of the 5 speech results contains the answer sentence
  out-of-scope   - queries with no answer should produce no *strong* match

Topic queries (a bare subject) count "on the second" only when the jump lands
where the explanation of that subject starts, not at a passing mention.

With --visual the slide & visual index is built too, and each query is also
scored as "speech + visuals": found if the speech result is on the second OR
the top "shown on screen" result lands inside one of the query's `screen`
ranges. `picture` queries describe an image and are scored on visuals only.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("VIDEORAG_DATA_DIR", os.path.join(HERE, ".cache"))
sys.path.insert(0, os.path.dirname(HERE))

from backend import jobs, processor, store  # noqa: E402  (env var must be set first)

ON_THE_SECOND = 5.0
SCREEN_TOLERANCE = 3.0


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


def _index(video_path: str, visual: bool) -> str:
    with open(video_path, "rb") as f:
        video_id = hashlib.md5(f.read()).hexdigest()[:16]
    need_audio = not store.is_indexed(video_id)
    need_visual = visual and not store.has_visual(video_id)
    if need_audio or need_visual:
        t0 = time.perf_counter()
        jobs.process_video(video_id, video_path, os.path.basename(video_path),
                           report=lambda stage, p: None, audio=need_audio, visual=need_visual)
        took = time.perf_counter() - t0
        print(f"Indexed ({'speech' if need_audio else ''}{' + ' if need_audio and need_visual else ''}"
              f"{'visuals' if need_visual else ''}) in {took/60:.1f} min")
    return video_id


def _on_screen(result: dict | None, ranges: list) -> bool:
    return bool(result) and any(lo - SCREEN_TOLERANCE <= result["jump_time"] <= hi + SCREEN_TOLERANCE
                                for lo, hi in ranges)


def main(video_path: str, queries_path: str, visual: bool) -> None:
    with open(queries_path, encoding="utf-8") as f:
        spec = json.load(f)
    video_id = _index(video_path, visual)
    with store._connect() as conn:
        sentences = [dict(r) for r in conn.execute(
            "SELECT start, end, text FROM sentences WHERE video_id = ? ORDER BY idx", (video_id,))]

    kinds = ("normal", "topic", "exact")
    tally = {k: {"second": 0, "passage": 0, "top5": 0, "combined": 0, "n": 0} for k in kinds}
    pictures = {"hit": 0, "n": 0}
    oos_ok = oos_total = 0
    latencies = []
    for item in spec["queries"]:
        kind = item.get("kind", "normal")
        t0 = time.perf_counter()
        results = store.search(video_id, item["q"], top_k=5)
        shown = store.search_visual(video_id, item["q"], top_k=4) if visual else []
        latencies.append(time.perf_counter() - t0)
        vtop = shown[0] if shown else None
        vdesc = (f" | screen {vtop['jump_time']:6.1f}s {vtop['source']:5s} {'strong' if vtop['strong'] else 'weak'}"
                 if vtop else (" | screen none" if visual else ""))

        if kind == "picture":
            if not visual:
                continue
            pictures["n"] += 1
            hit = _on_screen(vtop, item["screen"])
            pictures["hit"] += hit
            print(f"{'FOUND' if hit else 'MISS '} picture{vdesc}  {item['q']}")
            continue

        if item["answer"] is None:
            oos_total += 1
            ok = not any(r["strong"] for r in results + shown)
            oos_ok += ok
            print(f"{'OK   ' if ok else 'FAIL '} out-of-scope ({len(results)} speech, {len(shown)} screen)  {item['q']}")
            continue

        phrases = item["answer"] if isinstance(item["answer"], list) else [item["answer"]]
        starts = [a for a in (_answer_start(sentences, p) for p in phrases) if a is not None]
        row = tally[kind]
        row["n"] += 1
        if not starts:
            print(f"SKIP  answer phrase not in transcript: {item['answer']!r}")
            continue
        contains = lambda r: any(r["start"] <= a <= r["end"] for a in starts)  # noqa: E731
        top = results[0] if results else None
        on_sec = bool(top) and any(
            abs(top["jump_time"] - (a - store.LEAD_IN_SECONDS)) <= ON_THE_SECOND for a in starts)
        passage = bool(top) and contains(top)
        top5 = any(contains(r) for r in results)
        combined = on_sec or _on_screen(vtop, item.get("screen", []))
        row["second"] += on_sec; row["passage"] += passage; row["top5"] += top5; row["combined"] += combined
        label = "EXACT" if on_sec else ("NEAR " if passage else ("TOP5 " if top5 else "MISS "))
        if visual and combined and not on_sec:
            label = "SCRN "
        got = (f"{top['jump_time']:7.1f}s {'strong' if top['strong'] else 'weak  '}" if top else "   none       ")
        print(f"{label} answer {starts[0]:7.1f}s  speech {got}{vdesc}  {item['q']}")

    print()
    for kind, r in tally.items():
        if r["n"]:
            line = (f"{kind:7s} queries: on the second {r['second']}/{r['n']} | right passage {r['passage']}/{r['n']}"
                    f" | in top 5 {r['top5']}/{r['n']}")
            if visual:
                line += f" | speech + visuals {r['combined']}/{r['n']}"
            print(line)
    if pictures["n"]:
        print(f"picture queries: right frame first {pictures['hit']}/{pictures['n']}")
    print(f"out-of-scope: no strong match {oos_ok}/{oos_total}")
    warm = latencies[1:] or latencies
    print(f"search latency: {sum(warm)/len(warm):.2f}s avg (first query {latencies[0]:.1f}s incl. model load)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("queries")
    ap.add_argument("--visual", action="store_true", help="also build and score slide & visual search")
    a = ap.parse_args()
    main(a.video, a.queries, a.visual)

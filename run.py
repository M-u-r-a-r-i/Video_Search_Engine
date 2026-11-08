"""
Start VideoRAG and open it in your browser.

    python run.py                 # http://127.0.0.1:8000
    python run.py --port 9000
    python run.py --no-browser
"""

import argparse
import logging
import threading
import webbrowser

import uvicorn


def _warm_up_models() -> None:
    """Load the search models in the background so the first search is fast."""
    from backend import store
    try:
        store._get_embed_model()
        store._get_reranker()
        # CLIP is only needed (and only downloaded) once a lecture uses visual search.
        if any(v["visual"] for v in store.list_videos()):
            from backend import visual
            visual._get_clip()
        logging.getLogger("videorag").info("Search models loaded.")
    except Exception:
        logging.getLogger("videorag").exception("Could not preload search models")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the VideoRAG website locally.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="address to listen on (default: 127.0.0.1, this computer only)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from backend.api import create_app
    app = create_app()
    threading.Thread(target=_warm_up_models, daemon=True).start()

    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
    print(f"\n  VideoRAG is running at {url}\n  Press Ctrl+C to stop.\n")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

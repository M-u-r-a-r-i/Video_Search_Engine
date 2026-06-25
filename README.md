# 🎬 VideoRAG — Smart Video Search Engine

Upload a video and search **what is said and what is shown** using natural language,
then jump straight to the matching timestamp. VideoRAG transcribes the audio,
indexes both the speech and the visual frames, and runs multimodal semantic
search over them — all locally on your machine.

> **Version 1.0** — first public release. See [Roadmap](#-roadmap-v2) for what's next.

---

## ✨ Features

- **Speech search** — transcribes audio with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and indexes it in 30-second overlapping windows.
- **Visual search** — embeds video frames with OpenAI **CLIP** so you can find scenes by description.
- **Semantic retrieval** — bi-encoder recall (`all-mpnet-base-v2`) + cross-encoder reranking (`ms-marco-MiniLM-L-12-v2`) for relevance.
- **Jump-to-timestamp** — every result links to the exact moment in the player.
- **Runs locally** — no API keys, no cloud; your videos never leave your machine.
- **Polished Streamlit UI** — dark, single-page interface.

---

## 🧰 Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.10+** | 3.11/3.12 recommended. (3.14 works via a built-in compatibility shim.) |
| **FFmpeg** | Must be installed and on your `PATH` — used for audio + frame extraction. |
| ~2 GB free disk | Models are downloaded automatically on first run. |

### Installing FFmpeg

- **Windows:** `winget install Gyan.FFmpeg` (or download from [ffmpeg.org](https://ffmpeg.org/download.html))
- **macOS:** `brew install ffmpeg`
- **Linux (Debian/Ubuntu):** `sudo apt install ffmpeg`

Verify it's available:

```bash
ffmpeg -version
```

---

## 🚀 Setup & Usage

```bash
# 1. Clone the repository
git clone https://github.com/M-u-r-a-r-i/Video_Search_Engine.git
cd Video_Search_Engine

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**.

> ⏳ **First run** downloads the ML models (Whisper, MPNet, cross-encoder, CLIP).
> This can take a few minutes depending on your connection — subsequent runs are cached.

### How to use

1. **Drop a video** (`.mp4`, `.mkv`, `.webm`, `.mov`, `.avi`) onto the uploader.
2. Click **⚡ Process Video** — VideoRAG extracts audio, transcribes it, and indexes frames.
3. **Search** in natural language (e.g. *"when do they mention deadlines?"*).
4. Click a **▶ timestamp** to jump the player to that moment.

---

## 🗂️ Project Structure

```
Video_Search_Engine/
├── app.py                  # Streamlit frontend (UI, state, result cards)
├── backend/
│   ├── processor.py        # FFmpeg audio/frame extraction + Whisper transcription
│   └── vector_store.py     # Chunking, embeddings, ChromaDB storage & search
├── requirements.txt
└── README.md
```

Generated at runtime (git-ignored): `temp_data/` (extracted audio & frames) and
`chroma_db/` (the vector index).

---

## ⚙️ How It Works

```
Video ─► extract audio (ffmpeg) ─► transcribe (Whisper) ─► 30s sliding-window chunks
   │                                                              │
   └─► extract frames (1 / 5s) ─► CLIP image embeddings           └─► MPNet text embeddings
                     │                                                        │
                     └──────────────► ChromaDB (vector store) ◄──────────────┘
                                              │
   Query ─► MPNet + CLIP text encoders ─► retrieve ─► cross-encoder rerank ─► merge spans ─► results
```

---

## 🛠️ Tech Stack

Python · Streamlit · FFmpeg · faster-whisper · sentence-transformers ·
CLIP (transformers) · PyTorch · ChromaDB

---

## 🧭 Roadmap (v2)

Planned improvements for the next release:

- True text + visual score fusion per timestamp
- Single-pass frame extraction (major speed-up)
- Confidence-based "exact vs approximate" match labels
- Collection/temp cleanup on new video + cross-library search
- HTML-escaping and broader error logging

---

## 📄 License

Add a license of your choice (e.g. MIT) before sharing publicly.

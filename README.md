# 🎬 VideoRAG — Find the exact moment in any lecture

Upload a lecture, ask a question in plain English, and jump to **the exact second**
the answer is spoken. VideoRAG transcribes the speech, indexes every sentence, and
combines keyword and meaning-based search with a reranker, all locally on your
machine.

> *"when does the professor explain backpropagation?"* → **▶ 23:47**

---

## ✨ Features

- **Second-accurate results.** Whisper word timestamps let every result jump to the start of the sentence that answers your question, not to a 30-second block.
- **Understands questions, not just keywords.** Meaning-based search (`all-mpnet-base-v2`) finds answers even when you phrase things differently from the lecturer.
- **Never misses exact terms.** Keyword search (SQLite FTS5 / BM25) runs alongside, so names, formulas and jargon are always matched.
- **Reranked for relevance.** A cross-encoder (`ms-marco-MiniLM-L-12-v2`) picks the best passage and the best sentence inside it.
- **Honest confidence.** Results are labelled *strong* or *weak* instead of being silently hidden, so a correct answer never disappears.
- **Built for long lectures.** Uploads up to 4 GB, with live transcription progress.
- **Runs locally.** No API keys, no cloud; your videos never leave your machine.
- **Measured accuracy.** A benchmark script ships with the repo (see [Accuracy](#-accuracy)).

---

## 🧰 Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.10+** | Developed and tested on 3.14. |
| **FFmpeg** | Must be installed and on your `PATH`; used for audio extraction. |
| ~1.5 GB free disk | Models are downloaded automatically on first run. |

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

> ⏳ **First run** downloads the models (Whisper, MPNet, cross-encoder).
> This can take a few minutes depending on your connection; later runs use the cache.

### How to use

1. **Drop a lecture video** (`.mp4`, `.mkv`, `.webm`, `.mov`, `.avi`) onto the uploader.
2. Click **⚡ Process Video**. VideoRAG extracts the audio, transcribes it, and indexes every sentence.
   On a typical laptop CPU this runs at roughly **5× real time** (a 1-hour lecture takes about 12 minutes); a CUDA GPU is used automatically if available.
3. **Search** in natural language (e.g. *"when do they define eigenvalues?"*).
4. Click **▶ timestamp** to jump the player to that moment. The matching sentence is highlighted in each result.

Lectures you have processed before open instantly. Videos are identified by content, so renaming a file doesn't trigger reprocessing.

---

## 📊 Accuracy

Measured with [`benchmarks/run_benchmark.py`](benchmarks/run_benchmark.py) on an 18.5-minute
educational video, using 28 student-style questions plus 3 off-topic ones:

| | Result |
|---|---|
| Top result jumps to the answer sentence (±5 s) | **26 / 28** |
| Top result's passage contains the answer | **28 / 28** |
| Off-topic questions with no strong match | **3 / 3** |
| Search time (CPU) | ~1 s |
| Indexing speed (CPU, Whisper `base`) | ~5.6× real time |

Run it yourself on any video with your own question file:

```bash
python benchmarks/run_benchmark.py path/to/lecture.mp4 benchmarks/sugars_queries.json
```

**Known limitation:** terms Whisper mis-hears (rare names, non-English words) can only be found through
meaning-based search, and usually come back as *weak* matches.

---

## ⚙️ How It Works

```
Lecture ─► extract audio (ffmpeg) ─► Whisper (word timestamps) ─► sentences
                                                                     │
                                       overlapping sentence-aligned passages (~20 s)
                                                                     │
                                    SQLite: FTS5 keyword index + MPNet embeddings

Question ─► keyword (BM25) + meaning (MPNet) ─► fuse top 25 ─► cross-encoder rerank
         ─► pick best sentence in each passage ─► ▶ jump to that second (strong / weak)
```

---

## 🗂️ Project Structure

```
Video_Search_Engine/
├── app.py                    # Streamlit frontend (upload, progress, search, results)
├── backend/
│   ├── processor.py          # ffmpeg audio extraction + Whisper → timestamped sentences
│   ├── store.py              # SQLite index, hybrid search, reranking, sentence pinpointing
│   └── paths.py              # where runtime data lives
├── benchmarks/               # accuracy benchmark + question set
├── tests/                    # unit tests (pytest)
├── .streamlit/config.toml    # raises the upload limit for long lectures
└── requirements.txt
```

Runtime data is git-ignored and lives in `data/`: your uploaded videos (`data/videos/`) and the
search index (`data/library.sqlite3`). Set `VIDEORAG_DATA_DIR` to store it elsewhere.

---

## 🧪 Development

```bash
pip install -r requirements-dev.txt
pytest
```

The tests replace the ML models with simple stand-ins, so they run in seconds while still
exercising the real SQLite index and search pipeline.

---

## 🛠️ Tech Stack

Python · Streamlit · FFmpeg · faster-whisper · sentence-transformers · SQLite (FTS5) · NumPy

---

## 🧭 Roadmap

- YouTube links: paste a URL and use the existing captions (index in seconds)
- Search across a whole course of lectures at once
- Course-vocabulary hints so Whisper hears technical terms correctly
- Whisper model-size selector (accuracy vs speed)
- Search slide text (OCR on slide changes)

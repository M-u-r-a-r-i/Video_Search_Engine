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
- **Explanations, not just mentions.** Search for a topic like *"gradient descent"* and you land where the lecturer starts explaining it, shown as one section (e.g. *Explained here · 12:04 – 15:30*), with passing mentions labelled separately below.
- **Honest confidence.** Results are labelled *strong* or *weak* instead of being silently hidden, so a correct answer never disappears.
- **A real website, running on your PC.** Drag in lectures, watch them process live, and search beside the player. Clicking a result seeks instantly, and every search can be shared as a link to the exact moment.
- **Optional slide & visual search.** Turn it on for slide-heavy lectures and VideoRAG also reads text on slides and the board (OCR) and matches pictures by description (CLIP), e.g. *"Bayes theorem"* written on a slide or *"a graph of a sine wave"*. Off by default, so plain lectures stay fast.
- **Built for long lectures.** No upload size limit; lectures queue up and process in the background.
- **Runs locally.** No API keys, no cloud; your videos never leave your machine.
- **Measured accuracy.** A benchmark script ships with the repo (see [Accuracy](#-accuracy)).

---

## 🧰 Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.10+** | Developed and tested on 3.14. |
| **FFmpeg** | Must be installed and on your `PATH`; used for audio extraction. |
| ~1.5 GB free disk | Models are downloaded automatically on first run (+0.6 GB if you use visual search). |

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

# 4. Start VideoRAG
python run.py
```

Your browser opens **http://127.0.0.1:8000**. Press `Ctrl+C` in the terminal to stop.
Use `python run.py --port 9000` for another port, or `--no-browser` to skip opening a tab.

> ⏳ **First run** downloads the models (Whisper, MPNet, cross-encoder).
> This can take a few minutes depending on your connection; later runs use the cache.

### How to use

1. **Drop lecture videos** (`.mp4`, `.mkv`, `.webm`, `.mov`, `.avi`) onto the page. Processing starts automatically:
   VideoRAG extracts the audio, transcribes it, and indexes every sentence. On a typical laptop CPU this runs at roughly
   **5× real time** (a 1-hour lecture takes about 12 minutes); a CUDA GPU is used automatically if available.
2. **Open a lecture** and search in natural language: a topic (*"eigenvalues"*) to find where it is explained,
   or a question (*"why does gradient descent converge?"*) to jump to the answer. Press `/` to focus the search box.
3. **Click a result** to play from that moment. The matching sentence is highlighted.

### Slide & visual search (optional)

Speech isn't always enough: formulas, code, diagrams and labels are often shown but never read out.
Tick **"Also search slides & visuals"** before uploading, or click the button of the same name on a
lecture you already processed (speech is not re-transcribed). VideoRAG then:

- picks out **key frames** (whenever the picture visibly changes),
- reads their **on-screen text** with OCR ([RapidOCR](https://github.com/RapidAI/RapidOCR), bundled, offline),
- embeds them with **CLIP** so pictures can be found by description.

Matches appear under **"Shown on screen"** with a thumbnail of the frame. Speech search works while
the visual pass is still running. It is opt-in because it adds processing time: on a laptop CPU it took
**~160 s for an 18.5-minute video** (on top of ~200 s for speech), and less for lectures whose slides change rarely.

Everything you process stays in your library. Uploading the same lecture again (even renamed) reuses its index instantly.

---

## 📊 Accuracy

Measured with [`benchmarks/run_benchmark.py`](benchmarks/run_benchmark.py) on an 18.5-minute
educational video, using 28 student-style questions, 11 topic searches and 3 off-topic questions:

| | Result |
|---|---|
| Questions: top result jumps to the answer sentence (±5 s) | **26 / 28** |
| Topics: top result jumps to where the explanation starts (±5 s) | **11 / 11** |
| Questions: top result's section contains the answer | **28 / 28** |
| Off-topic questions with no strong match | **3 / 3** |
| Search time (CPU) | ~1 s |
| Indexing speed (CPU, Whisper `base`) | ~5.6× real time |

Run it yourself on any video with your own question file:

```bash
python benchmarks/run_benchmark.py path/to/lecture.mp4 benchmarks/sugars_queries.json
python benchmarks/run_benchmark.py path/to/lecture.mp4 benchmarks/universe_queries.json --visual
```

**Known limitation:** terms Whisper mis-hears (rare names, non-English words) can only be found through
meaning-based search, and usually come back as *weak* matches. If they appear on screen, visual search
finds them reliably: *"aguamiel"*, misheard by Whisper, is found on the slide where it is written.

Visual search on the same video (top result checked by eye):

| | Result |
|---|---|
| Words shown on screen (labels, captions, map names) | **6 / 6** |
| Pictures by description | **4 / 6** (CLIP can't read labels: *"a bottle of corn syrup"* found an agave syrup bottle) |
| Things not in the video (*"a laptop computer"*) | **2 / 2** return nothing |

**Second video:** an 18.6-minute physics explainer (gravitons and gravitational-wave detectors), denser
and closer to a real lecture. Its benchmark is [`benchmarks/universe_queries.json`](benchmarks/universe_queries.json);
"speech + visuals" counts a query as found if either the speech result or the top on-screen result lands on the answer.

| | Speech only | Speech + visuals |
|---|---|---|
| Questions (12) | 9 | **10** |
| Topic searches (4) | 1 | **2** |
| Exact terms (6) | 2 | **5** |
| Pictures by description (9) | n/a | **7** |
| Off-topic, no strong match (3) | 3 | **3** |

Visual search matters most for technical terms Whisper mishears: it said *"burillion bar"*, *"neobium"* and
*"phone on detector"*, while the slides spell *Beryllium*, *Niobium* and *Phonon detector* correctly.
Topic searches are the weak spot on dense lectures: when a lecturer explains something for minutes without
naming it again, VideoRAG may jump to a later recap instead of the start of the explanation.

---

## ⚙️ How It Works

```
Lecture ─► extract audio (ffmpeg) ─► Whisper (word timestamps) ─► sentences
                                                                     │
                                       overlapping sentence-aligned passages (~20 s)
                                                                     │
                                    SQLite: FTS5 keyword index + MPNet embeddings

Query ─► keyword (BM25) + meaning (MPNet) ─► fuse top 25 ─► cross-encoder rerank
      ─► group neighbouring passages into sections (explained vs. mention)
      ─► topic: jump to where the section starts · question: jump to the answering sentence

Optional visual pass:
Lecture ─► a frame every 2 s ─► keep key frames (visible change) ─► OCR text + CLIP embedding
Query   ─► on-screen text: keyword + meaning ─► rerank   |   pictures: CLIP similarity, kept
           only if a frame clearly stands out from the rest of the video ─► "Shown on screen"
```

The website is a small FastAPI server (`backend/api.py`) with a JSON API and a plain HTML/CSS/JS
frontend (`web/`, no build step). Videos are processed one at a time on a background worker
(`backend/jobs.py`) while the page polls their progress. API docs: http://127.0.0.1:8000/api/docs

---

## 🗂️ Project Structure

```
Video_Search_Engine/
├── run.py                    # start the website: python run.py
├── web/                      # frontend: index.html, app.js, styles.css (no build step)
├── backend/
│   ├── api.py                # FastAPI server: upload, library, video streaming, search
│   ├── jobs.py               # background processing queue with progress
│   ├── processor.py          # ffmpeg audio extraction + Whisper → timestamped sentences
│   ├── visual.py             # optional: key frames, OCR, CLIP embeddings
│   ├── store.py              # SQLite index, hybrid search, reranking, sections, pinpointing
│   └── paths.py              # where runtime data lives
├── benchmarks/               # accuracy benchmark + question set
├── tests/                    # unit + API tests (pytest)
└── requirements.txt
```

Runtime data is git-ignored and lives in `data/`: your uploaded videos (`data/videos/`) and the
search index (`data/library.sqlite3`), plus key-frame thumbnails (`data/frames/`) for lectures using
visual search. Set `VIDEORAG_DATA_DIR` to store it elsewhere.

---

## 🧪 Development

```bash
pip install -r requirements-dev.txt
pytest
```

The tests replace the ML models and Whisper with simple stand-ins, so they run in seconds while
still exercising the real SQLite index, search pipeline and HTTP API.

---

## 🛠️ Tech Stack

Python · FastAPI · FFmpeg · faster-whisper · sentence-transformers · CLIP · RapidOCR · SQLite (FTS5) · NumPy · vanilla JS

---

## 🧭 Roadmap

- YouTube links: paste a URL and use the existing captions (index in seconds)
- Search across a whole course of lectures at once
- Course-vocabulary hints so Whisper hears technical terms correctly
- Whisper model-size selector (accuracy vs speed)

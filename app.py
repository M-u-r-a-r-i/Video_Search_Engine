"""
app.py
Streamlit frontend for the VideoRAG application — Dark premium, no sidebar.
"""

import os
import hashlib
import streamlit as st

from backend.processor import extract_audio, transcribe_audio, extract_frames, TEMP_DIR
from backend.vector_store import add_video, add_video_frames, search

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Video Search",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — Dark Premium, No Sidebar
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ===================== KEYFRAME ANIMATIONS ===================== */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 12px rgba(139,92,246,0.2); }
    50%      { box-shadow: 0 0 28px rgba(139,92,246,0.45); }
}
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(60px); }
    to   { opacity: 1; transform: translateX(0); }
}

/* ===================== GLOBAL ===================== */
html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
    color: #E0E0E0;
}
.stApp {
    background: linear-gradient(160deg, #0A0A0F 0%, #12121F 35%, #0E0E1A 70%, #0A0A12 100%);
    min-height: 100vh;
}
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1100px;
}

/* ===================== NUKE EVERYTHING STREAMLIT ===================== */
header[data-testid="stHeader"] { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stStatusWidget"] { border-radius: 14px !important; }

/* Force-kill sidebar completely */
section[data-testid="stSidebar"] {
    display: none !important;
    width: 0 !important;
    min-width: 0 !important;
    max-width: 0 !important;
    visibility: hidden !important;
    overflow: hidden !important;
}
[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
button[kind="headerNoPadding"] { display: none !important; }

/* ===== NUCLEAR: Override Material Symbols font to be invisible ===== */
/* This catches ALL icon text no matter where Streamlit renders it */
@font-face {
    font-family: 'Material Symbols Rounded';
    src: local('__empty__');
    font-display: block;
}
@font-face {
    font-family: 'Material Symbols Outlined';
    src: local('__empty__');
    font-display: block;
}
@font-face {
    font-family: 'Material Icons';
    src: local('__empty__');
    font-display: block;
}
/* Also hide by class just in case */
[class*="material-symbols"],
[class*="material-icons"] {
    display: none !important;
}
/* Hide icon text inside buttons and status widgets */
[data-testid="stBaseButton-headerNoPadding"],
[data-testid="stBaseButton-header"] {
    display: none !important;
}
/* Make any leftover icon-font spans invisible */
span[style*="font-family"] {
    font-size: 0 !important;
    color: transparent !important;
    -webkit-text-fill-color: transparent !important;
    width: 0 !important;
    overflow: hidden !important;
}

/* ===================== HERO ===================== */
.hero-wrapper {
    text-align: center;
    animation: fadeInUp 0.7s ease-out both;
    padding-top: 1rem;
}
.hero-heading {
    display: inline-block;
    font-weight: 800;
    font-size: 2.8rem;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #A78BFA, #818CF8, #C084FC, #A78BFA);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 6s ease infinite;
}
.hero-sub {
    color: #52525B;
    font-weight: 400;
    font-size: 1rem;
    margin-top: 0.3rem;
}
.hero-divider {
    width: 50px;
    height: 3px;
    background: linear-gradient(90deg, #A78BFA, #818CF8);
    border-radius: 4px;
    margin: 1rem auto 1.5rem auto;
    animation: fadeIn 1s ease-out 0.3s both;
}

/* ===================== SEARCH BAR ===================== */
.stTextInput > label { display: none !important; }
.stTextInput > div > div > input {
    border-radius: 14px !important;
    border: 1px solid rgba(139,92,246,0.2) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.03) !important;
    padding: 0.95rem 1.3rem !important;
    font-size: 1rem !important;
    color: #E4E4E7 !important;
    background: rgba(22,22,36,0.75) !important;
    backdrop-filter: blur(12px) !important;
    transition: all 0.3s ease !important;
}
.stTextInput > div > div > input::placeholder { color: #52525B !important; }
.stTextInput > div > div > input:focus {
    border-color: #8B5CF6 !important;
    box-shadow: 0 4px 28px rgba(139,92,246,0.2), 0 0 0 3px rgba(139,92,246,0.08) !important;
}

/* ===================== BUTTONS ===================== */
.stButton > button {
    border-radius: 12px;
    font-weight: 600;
    padding: 0.6rem 1.4rem;
    background: linear-gradient(135deg, #7C3AED 0%, #6D28D9 100%);
    color: #FFFFFF !important;
    border: none;
    box-shadow: 0 4px 16px rgba(124,58,237,0.3);
    transition: all 0.25s cubic-bezier(.4,0,.2,1);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(124,58,237,0.4);
    background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%);
}

/* ===================== FILE UPLOADER ===================== */
[data-testid="stFileUploader"] section {
    border: 2px dashed rgba(139,92,246,0.18) !important;
    border-radius: 14px !important;
    background: rgba(20,20,32,0.4) !important;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"] section:hover {
    border-color: rgba(139,92,246,0.4) !important;
    background: rgba(28,28,44,0.4) !important;
}
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small { color: #71717A !important; }

/* ===================== VIDEO ===================== */
video {
    border-radius: 14px !important;
    box-shadow: 0 8px 36px rgba(0,0,0,0.5);
}

/* ===================== RESULT CARDS ===================== */
.result-card {
    background: rgba(24,24,38,0.65);
    backdrop-filter: blur(16px);
    border-radius: 14px;
    border: 1px solid rgba(139,92,246,0.08);
    border-left: 3px solid #8B5CF6;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.7rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.25);
    transition: all 0.22s cubic-bezier(.4,0,.2,1);
    cursor: pointer;
    animation: fadeInUp 0.4s ease-out both;
}
.result-card:hover {
    transform: translateY(-3px) scale(1.005);
    box-shadow: 0 8px 28px rgba(139,92,246,0.12);
    border-color: rgba(139,92,246,0.25);
    border-left-color: #A78BFA;
}
.result-card .timestamp {
    font-size: 0.72rem;
    font-weight: 700;
    color: #A78BFA;
    margin-bottom: 0.35rem;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}
.result-card .excerpt {
    font-size: 0.85rem;
    color: #A1A1AA;
    line-height: 1.6;
}
.result-card .score-badge {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 600;
    color: #C4B5FD;
    background: rgba(139,92,246,0.1);
    border: 1px solid rgba(139,92,246,0.15);
    border-radius: 8px;
    padding: 0.15rem 0.5rem;
    margin-top: 0.5rem;
}

/* ===================== RESULTS PANEL ===================== */
.results-panel-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #A78BFA;
    margin-bottom: 0.3rem;
}
.results-panel-query {
    font-size: 1rem;
    font-weight: 600;
    color: #E4E4E7;
    margin-bottom: 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(139,92,246,0.1);
}

/* ===================== BADGES ===================== */
.active-video-badge {
    display: inline-block;
    background: rgba(34,197,94,0.08);
    color: #4ADE80;
    font-weight: 600;
    font-size: 0.78rem;
    padding: 0.3rem 0.85rem;
    border-radius: 10px;
    border: 1px solid rgba(34,197,94,0.12);
    margin-bottom: 1rem;
    animation: fadeIn 0.5s ease-out both;
}

/* ===================== SECTION LABEL ===================== */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    color: #3F3F46;
    margin-bottom: 0.75rem;
}

/* ===================== EMPTY STATE ===================== */
.empty-state {
    text-align: center;
    margin-top: 6rem;
    animation: fadeInUp 0.8s ease-out both;
}
.empty-icon {
    font-size: 3.5rem;
    margin-bottom: 1rem;
    display: inline-block;
    animation: pulseGlow 3s ease-in-out infinite;
    background: rgba(139,92,246,0.06);
    border-radius: 20px;
    padding: 1rem 1.4rem;
}
.empty-msg {
    color: #52525B;
    font-size: 0.95rem;
}
.empty-msg strong { color: #A78BFA; }

/* ===================== SCROLLBAR ===================== */
/* ===================== FALLBACK BANNER ===================== */
.fallback-banner {
    background: rgba(234, 179, 8, 0.08);
    border: 1px solid rgba(234, 179, 8, 0.2);
    border-left: 3px solid #EAB308;
    border-radius: 12px;
    padding: 0.85rem 1.2rem;
    margin-bottom: 1rem;
    animation: fadeIn 0.5s ease-out both;
}
.fallback-banner .fb-icon { font-size: 1.1rem; margin-right: 0.4rem; }
.fallback-banner .fb-title {
    font-weight: 700;
    font-size: 0.82rem;
    color: #EAB308;
}
.fallback-banner .fb-msg {
    font-size: 0.8rem;
    color: #A1A1AA;
    margin-top: 0.3rem;
    line-height: 1.5;
}

/* ===================== NO RESULTS ===================== */
.no-results {
    text-align: center;
    padding: 2rem 1rem;
    animation: fadeInUp 0.5s ease-out both;
}
.no-results .nr-icon { font-size: 2.5rem; margin-bottom: 0.6rem; }
.no-results .nr-msg {
    color: #71717A;
    font-size: 0.9rem;
}
.no-results .nr-msg strong { color: #A78BFA; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.2); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: rgba(139,92,246,0.4); }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "video_id": None,
        "video_path": None,
        "processed": False,
        "seek_time": 0,
        "results": [],
        "match_type": "",
        "last_query": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ---------------------------------------------------------------------------
# Helper: format seconds -> mm:ss
# ---------------------------------------------------------------------------
def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Helper: build a single result card HTML (no indentation!)
# ---------------------------------------------------------------------------
def _card_html(idx: int, ts_label: str, excerpt: str, score: float) -> str:
    return (
        f'<div class="result-card" style="animation-delay:{idx*0.06}s">'
        f'<div class="timestamp">⏱ {ts_label}</div>'
        f'<div class="excerpt">{excerpt}</div>'
        f'<span class="score-badge">relevance: {score:.2f}</span>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# UI — Hero
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="hero-wrapper">'
    '<div class="hero-heading">🎬 VideoRAG</div>'
    '<p class="hero-sub">Upload a video and search its content with natural language.</p>'
    '<div class="hero-divider"></div>'
    '</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# STATE: No video loaded yet — show upload
# ---------------------------------------------------------------------------
if not st.session_state.video_path or not st.session_state.processed:

    uploaded = st.file_uploader(
        "Drop a video file here",
        type=["mp4", "mkv", "webm", "mov", "avi"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        os.makedirs(TEMP_DIR, exist_ok=True)
        save_path = os.path.join(TEMP_DIR, uploaded.name)

        if st.session_state.video_path != save_path:
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.session_state.video_path = save_path
            st.session_state.video_id = hashlib.md5(
                uploaded.name.encode()
            ).hexdigest()[:12]
            st.session_state.processed = False
            st.session_state.results = []
            st.session_state.seek_time = 0
            st.session_state.last_query = ""

        if not st.session_state.processed:
            st.markdown("<br>", unsafe_allow_html=True)
            _l, col_btn, _r = st.columns([1, 2, 1])
            with col_btn:
                if st.button("⚡ Process Video", use_container_width=True):
                    with st.status("Processing video…", expanded=True) as status:
                        st.write("🎵 Extracting audio…")
                        audio_path = extract_audio(save_path)
                        st.write("🗣️ Transcribing speech…")
                        segments = transcribe_audio(audio_path)
                        st.write(f"📦 Embedding {len(segments)} segments…")
                        n_chunks = add_video(st.session_state.video_id, segments)
                        st.write("🖼️ Extracting video frames…")
                        frames = extract_frames(
                            save_path, st.session_state.video_id
                        )
                        n_frames = 0
                        if frames:
                            st.write(f"🧠 Embedding {len(frames)} frames with CLIP…")
                            n_frames = add_video_frames(
                                st.session_state.video_id, frames
                            )
                        status.update(
                            label=f"✅ Done — {n_chunks} text chunks + {n_frames} frames indexed",
                            state="complete",
                        )
                    st.session_state.processed = True
                    st.rerun()
    else:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-icon">📂</div>'
            '<p class="empty-msg">Drop a <strong>video file</strong> above to get started.</p>'
            '</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# STATE: Video processed — player + search + results
# ---------------------------------------------------------------------------
else:
    # Badge
    fname = os.path.basename(st.session_state.video_path)
    st.markdown(
        f'<div class="active-video-badge">📹 {fname}</div>',
        unsafe_allow_html=True,
    )

    # Video player — full width
    st.video(st.session_state.video_path, start_time=int(st.session_state.seek_time))

    # Search bar
    st.markdown("<br>", unsafe_allow_html=True)
    _sl, col_search, _sr = st.columns([0.5, 5, 0.5])
    with col_search:
        query = st.text_input(
            "Search",
            placeholder="e.g. 'When does the speaker mention deadlines?'",
            label_visibility="collapsed",
        )

    # Run search
    if query and query != st.session_state.last_query:
        with st.spinner("🔍 Searching video content…"):
            result = search(
                st.session_state.video_id, query, top_k=5
            )
        st.session_state.results = result.get("results", [])
        st.session_state.match_type = result.get("match_type", "")
        st.session_state.last_query = query

    # ---------- Show results ----------
    if st.session_state.last_query and st.session_state.match_type == "none":
        # No results at all
        st.markdown(
            '<div class="no-results">'
            '<div class="nr-icon">🔎</div>'
            '<p class="nr-msg">No matching content found for '
            f'<strong>"{st.session_state.last_query}"</strong>. '
            'Try a different query.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    elif st.session_state.results:
        st.markdown("<br>", unsafe_allow_html=True)

        # Fallback banner (keyword-only matches)
        if st.session_state.match_type == "keyword":
            st.markdown(
                '<div class="fallback-banner">'
                '<span class="fb-icon">⚠️</span>'
                '<span class="fb-title">Approximate matches</span>'
                '<p class="fb-msg">I couldn\'t find the exact context you asked for. '
                'Here are the closest timestamps where related keywords appear:</p>'
                '</div>',
                unsafe_allow_html=True,
            )

        # Results header
        st.markdown(
            '<p class="results-panel-title">Search Results</p>'
            f'<p class="results-panel-query">"{st.session_state.last_query}"</p>',
            unsafe_allow_html=True,
        )

        # Result cards — rendered as HTML
        cards = ""
        for idx, r in enumerate(st.session_state.results):
            ts = f"{_fmt(r['start_time'])} → {_fmt(r['end_time'])}"
            txt = r["text"][:200] + ("…" if len(r["text"]) > 200 else "")
            cards += _card_html(idx, ts, txt, r["score"])

        st.markdown(cards, unsafe_allow_html=True)

        # Jump-to-timestamp buttons
        st.markdown(
            '<p class="section-label" style="margin-top:1rem;">Jump to timestamp</p>',
            unsafe_allow_html=True,
        )
        btn_cols = st.columns(min(len(st.session_state.results), 5))
        for idx, r in enumerate(st.session_state.results):
            ts = f"{_fmt(r['start_time'])}–{_fmt(r['end_time'])}"
            with btn_cols[idx]:
                if st.button(f"▶ {ts}", key=f"seek_{idx}"):
                    st.session_state.seek_time = r["start_time"]
                    st.rerun()

    # New video button
    st.markdown("<br>", unsafe_allow_html=True)
    _bl, col_new, _br = st.columns([2, 1, 2])
    with col_new:
        if st.button("🔄 New Video", use_container_width=True):
            for k in ("video_path", "video_id", "processed", "results", "match_type", "seek_time", "last_query"):
                st.session_state[k] = None if k in ("video_path", "video_id") else (
                    False if k == "processed" else (
                        [] if k == "results" else (
                            0 if k == "seek_time" else ""
                        )
                    )
                )
            st.rerun()

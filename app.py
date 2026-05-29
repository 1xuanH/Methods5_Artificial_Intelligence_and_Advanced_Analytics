import os
import tempfile
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import matplotlib.pyplot as plt

from emotion_app import predict_emotion, transcribe_audio

warnings.filterwarnings("ignore", message="`return_all_scores` is now deprecated")
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

st.set_page_config(page_title="Emotion Release Emoji Machine", page_icon="🤖", layout="centered")

# -------------------------
# Load CSS
# -------------------------
def load_css(path="style.css"):
    css = ""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            css = f.read()

    hard_override = """
    <style>
      html, body { height: 100%; }

      .stApp, .stAppViewContainer, .stAppViewContainer > .main,
      .stAppViewContainer > .main > div,
      section.main, main, [data-testid="stAppViewContainer"],
      [data-testid="stHeader"], [data-testid="stToolbar"] {
        background: transparent !important;
      }

      .block-container, [data-testid="block-container"]{
        background: transparent !important;
      }

      [data-testid="stAppViewContainer"] .main {
        padding-top: 0rem !important;
      }
    </style>
    """
    st.markdown(hard_override, unsafe_allow_html=True)

    if css.strip():
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css()

# -------------------------
# Emoji floating background
# -------------------------
def render_emoji_bg(n=48):
    emojis = [
        "💗","🦄","🫧","🌈","😊","🌷","💜","⭐️","😌","🧋",
        "🩵","🧸","💚","🍰","🎨","🏖️","🌸","☁️","⛄️","🐶"
    ]
    spans = []
    for i in range(n):
        spans.append(f"<span class='e'>{emojis[i % len(emojis)]}</span>")

    st.markdown(
        f"<div class='emoji-rain' aria-hidden='true'>{''.join(spans)}</div>",
        unsafe_allow_html=True
    )

render_emoji_bg()

# -------------------------
# Fancy divider
# -------------------------
def fancy_divider():
    st.markdown("<div class='fancy-divider' aria-hidden='true'></div>", unsafe_allow_html=True)

# -------------------------
# Progress card HTML
# -------------------------
def progress_bar_html(is_running: bool) -> str:
    cls = "is-running" if is_running else "is-idle"
    return f"""
    <div class="loading-card {cls}" aria-hidden="true">
      <div class="loading-top">
        <div class="loading-label">Model loading / running</div>
        <div class="loading-dots"><i></i><i></i><i></i></div>
      </div>
      <div class="loading-track">
        <div class="loading-fill"></div>
        <div class="loading-shine"></div>
      </div>
    </div>
    """

# -------------------------
# Session state
# -------------------------
if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "shake" not in st.session_state:
    st.session_state.shake = False

if "is_running" not in st.session_state:
    st.session_state.is_running = False

if "feedback_up" not in st.session_state:
    st.session_state.feedback_up = 0
if "feedback_down" not in st.session_state:
    st.session_state.feedback_down = 0

# -------------------------
# HERO
# -------------------------
hero_html = """
<div class="hero-new">

  <div class="hero-emojis" aria-hidden="true">
    <span>😈</span><span>💐</span><span>❤️‍🔥</span><span>🎠</span>
    <span>😻</span><span>🤡</span><span>👻</span><span>👑</span>
  </div>

  <div class="hero-new-inner">
    <span class="hero-new-emoji hero-new-left" aria-hidden="true">🤖</span>
    <span class="hero-new-title hero-new-center">Emotion Release Emoji Machine</span>
    <span class="hero-new-emoji hero-new-right" aria-hidden="true">🎀</span>
  </div>

  <p class="hero-new-sub">🎤 Upload audio or ⌨️ Enter text — I’ll analyze your mood with emojis 🫣</p>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

fancy_divider()

# -------------------------
# Result box
# -------------------------
def render_result_box():
    r = st.session_state.last_result
    if not r:
        return

    shake_class = "shake" if st.session_state.shake else ""
    st.session_state.shake = False

    st.markdown(
        f"""
        <div class="result {shake_class}">
          <div class="result-emoji">{r["emoji"]}</div>
          <div class="result-emotion">{r["emotion"].upper()} 🌤️</div>
          <div class="result-meta">Confidence: {r["confidence"]:.1%}</div>
          <div class="result-feedback">{r["feedback"]}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------
# Mode selection
# -------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">🤔 Choose input mode</div>', unsafe_allow_html=True)

mode = st.radio(
    "Input mode",
    ["Text ✍️", "Audio 🎙️"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("</div>", unsafe_allow_html=True)

fancy_divider()

# -------------------------
# Text mode
# -------------------------
if mode == "Text ✍️":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">✍️ Text input</div>', unsafe_allow_html=True)

    text_input = st.text_area(
        "Text",
        placeholder="e.g., I’m excited but a bit nervous today...",
        label_visibility="collapsed"
    )

    run_text = st.button("🔮 Start Recognition & Analysis 👾", use_container_width=True)

    loader_slot = st.empty()
    loader_slot.markdown(progress_bar_html(st.session_state.is_running), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if run_text:
        if not text_input or not text_input.strip():
            st.warning("Please type some text first 🥺")
        else:
            st.session_state.is_running = True
            loader_slot.markdown(progress_bar_html(True), unsafe_allow_html=True)

            with st.spinner("Reading your emotion... 💫"):
                emotion, confidence, scores, emoji, feedback = predict_emotion(text_input.strip())

            st.session_state.last_result = {
                "emotion": emotion,
                "confidence": confidence,
                "scores": scores,
                "emoji": emoji,
                "feedback": feedback,
                "transcription": None
            }

            st.session_state.is_running = False
            st.rerun()

# -------------------------
# Audio mode
# -------------------------
else:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🎙️ Audio input</div>', unsafe_allow_html=True)

    audio_file = st.file_uploader(
        "Upload an audio file",
        type=["wav", "mp3", "m4a"]
    )

    run_audio = st.button("🗣️ Start Transcription & Analysis 🎵", use_container_width=True)

    loader_slot = st.empty()
    loader_slot.markdown(progress_bar_html(st.session_state.is_running), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if run_audio:
        if not audio_file:
            st.warning("Please upload an audio file first 🥺")
        else:
            suffix = os.path.splitext(audio_file.name)[1].lower() or ".wav"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name

            try:
                st.session_state.is_running = True
                loader_slot.markdown(progress_bar_html(True), unsafe_allow_html=True)

                with st.spinner("Listening carefully... 🎧🎶"):
                    text = transcribe_audio(tmp_path)
                    emotion, confidence, scores, emoji, feedback = predict_emotion(text)

                st.session_state.last_result = {
                    "emotion": emotion,
                    "confidence": confidence,
                    "scores": scores,
                    "emoji": emoji,
                    "feedback": feedback,
                    "transcription": text
                }

                st.session_state.is_running = False
                st.rerun()

            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

# -------------------------
# Render results
# -------------------------
render_result_box()

if st.session_state.last_result:
    fancy_divider()

# -------------------------
# Release / charts / feedback
# -------------------------
def render_release_and_chart():
    r = st.session_state.last_result
    if not r:
        return

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🫠 Emotion release button</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            "<div style='color:#6b7280;font-weight:600;margin-bottom:6px;'>"
            "Press it to 'SHAKE IT OFF' 😤🚫💣</div>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown('<div class="release-btn">', unsafe_allow_html=True)
        if st.button("💥 Release!", use_container_width=True):
            st.session_state.shake = True
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if r.get("transcription"):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📝 Transcription</div>', unsafe_allow_html=True)
        st.write(r["transcription"])
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("## 📊 Emotion Confidence Scores")

    fig, ax = plt.subplots()
    ax.bar(list(r["scores"].keys()), list(r["scores"].values()))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Emotion Confidence Scores")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig)

    st.markdown(
        f"""
        <div class="fb-bar">
          <div class="fb-left">
            <span class="fb-icon">📈</span>
            <span class="fb-text">Feedback received:</span>
            <span class="fb-pill up">👍 {st.session_state.feedback_up}</span>
            <span class="fb-pill down">👎 {st.session_state.feedback_down}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    b1, b2, b3 = st.columns([1, 1, 1])

    with b1:
        if st.button("👍 Helpful", use_container_width=True, key="fb_up"):
            st.session_state.feedback_up += 1
            st.rerun()

    with b2:
        if st.button("👎 Wrong", use_container_width=True, key="fb_down"):
            st.session_state.feedback_down += 1
            st.rerun()

    with b3:
        if st.button("🔁 Reset", use_container_width=True, key="fb_reset"):
            st.session_state.feedback_up = 0
            st.session_state.feedback_down = 0
            st.rerun()

render_release_and_chart()

st.markdown(
    """
    <div style="margin-top:10px; color:#6b7280; line-height:1.6; text-align:center;">
      <div style="display:inline-block; text-align:left; max-width:720px;">
        <strong>Tip:</strong><br>
        Shorter audio clips with clear pronunciation usually result in more accurate transcriptions.✨<br>
        Text with stronger emotional expression can also make recognition and analysis more accurate and efficient.💪🏻
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

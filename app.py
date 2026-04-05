import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import plotly.graph_objects as go
import os

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Emotion AI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ------------------ CUSTOM STYLE ------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #060b14;
        color: #e2e8f0;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 800px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .hero-wrap {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #38bdf8;
        font-size: 12px;
        font-weight: 500;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 5px 14px;
        border-radius: 20px;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: clamp(2.2rem, 5vw, 3.2rem);
        font-weight: 800;
        line-height: 1.1;
        background: linear-gradient(135deg, #f0f9ff 30%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.6rem;
    }
    .hero-sub {
        font-size: 1rem;
        color: #64748b;
        font-weight: 300;
        letter-spacing: 0.3px;
    }

    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #1e3a5f, transparent);
        margin: 1.5rem 0;
    }

    [data-testid="stFileUploader"] {
        background: rgba(14, 30, 54, 0.6);
        border: 1.5px dashed rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 1.5rem;
        transition: border-color 0.3s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(56, 189, 248, 0.55);
    }

    [data-testid="stImage"] img {
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }

    .result-card {
        background: rgba(14, 30, 54, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1rem;
    }
    .emotion-label {
        font-family: 'Syne', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        color: #38bdf8;
        line-height: 1;
        margin-bottom: 0.3rem;
    }
    .confidence-val {
        font-size: 1rem;
        color: #94a3b8;
        font-weight: 400;
    }
    .confidence-val span {
        color: #7dd3fc;
        font-weight: 600;
        font-size: 1.15rem;
    }

    .conf-bar-bg {
        background: rgba(255,255,255,0.06);
        border-radius: 999px;
        height: 8px;
        margin-top: 1rem;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.8s ease;
    }

    .emotion-icon { font-size: 2rem; margin-bottom: 0.4rem; }

    .section-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 0.8rem;
    }

    [data-testid="stAlert"] { border-radius: 12px; }
    [data-testid="stSpinner"] { color: #38bdf8; }
    </style>
""", unsafe_allow_html=True)


# ------------------ CONSTANTS ------------------
# ⚠️ ORDER MATTERS: image_dataset_from_directory assigns labels alphabetically.
# FER dataset folders sorted alphabetically:
# angry(0), disgust(1), fear(2), happy(3), neutral(4), sad(5), surprise(6)
CLASS_NAMES  = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
CLASS_EMOJIS = ["😠",    "🤢",      "😨",   "😄",    "😐",      "😢",  "😲"]
CLASS_COLORS = ["#ef4444","#a855f7","#f97316","#22c55e","#94a3b8","#60a5fa","#f59e0b"]


# ------------------ LOAD MODEL ------------------
@st.cache_resource(show_spinner=False)
def load_my_model():
    model_path = "emotion_model.keras"
    if not os.path.exists(model_path):
        st.error("❌ `emotion_model.keras` not found. Make sure it is committed to your repo.")
        st.stop()
    try:
        return load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        st.stop()

with st.spinner("Loading model…"):
    model = load_my_model()


# ------------------ HERO ------------------
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">✦ Neural Emotion Recognition</div>
    <div class="hero-title">Emotion Detection AI</div>
    <div class="hero-sub">Upload a face image — the model reads the emotion in milliseconds</div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)


# ------------------ UPLOAD ------------------
uploaded_file = st.file_uploader(
    "Drop an image here, or click to browse",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)
st.markdown(
    '<p style="text-align:center;color:#334155;font-size:13px;margin-top:0.4rem;">'
    'Supported formats: JPG · PNG · WEBP</p>',
    unsafe_allow_html=True
)


# ------------------ INFERENCE ------------------
if uploaded_file is not None:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("Could not open the image. Please try a different file.")
        st.stop()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<p class="section-label">Input Image</p>', unsafe_allow_html=True)
        st.image(image, use_column_width=True)

    # --- Preprocessing: resize to 224x224 RGB to match ResNet50 training ---
    with st.spinner("Analyzing emotion…"):
        img     = image.resize((224, 224))
        img_arr = np.array(img, dtype=np.float32)

        # Apply ResNet50 preprocess_input (same as training pipeline)
        from tensorflow.keras.applications.resnet import preprocess_input
        img_arr = preprocess_input(img_arr)
        img_arr = np.expand_dims(img_arr, axis=0)

        prediction = model.predict(img_arr, verbose=0)[0]

    pred_idx   = int(np.argmax(prediction))
    confidence = float(prediction[pred_idx])
    pred_label = CLASS_NAMES[pred_idx]
    pred_emoji = CLASS_EMOJIS[pred_idx]
    pred_color = CLASS_COLORS[pred_idx]

    with col2:
        st.markdown('<p class="section-label">Result</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="result-card">
            <div class="emotion-icon">{pred_emoji}</div>
            <div class="emotion-label">{pred_label}</div>
            <div class="confidence-val">Confidence: <span>{confidence:.1%}</span></div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill"
                     style="width:{confidence*100:.1f}%;
                            background: linear-gradient(90deg, {pred_color}99, {pred_color});">
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        sorted_idx  = np.argsort(prediction)[::-1]
        second_idx  = int(sorted_idx[1])
        second_conf = float(prediction[second_idx])
        if second_conf > 0.12:
            st.markdown(
                f'<p style="color:#475569;font-size:13px;margin-top:-0.4rem;">'
                f'Runner-up: {CLASS_EMOJIS[second_idx]} '
                f'<strong style="color:#64748b">{CLASS_NAMES[second_idx]}</strong> '
                f'<span style="color:#334155">({second_conf:.1%})</span></p>',
                unsafe_allow_html=True
            )

    # --- Probability Chart ---
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-label" style="text-align:center">All Emotion Probabilities</p>',
        unsafe_allow_html=True
    )

    fig = go.Figure(go.Bar(
        x=CLASS_NAMES,
        y=[float(p) for p in prediction],
        marker=dict(
            color=[CLASS_COLORS[i] if i == pred_idx else "#1e3a5f" for i in range(len(CLASS_NAMES))],
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        text=[f"{p:.1%}" for p in prediction],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=12, family="DM Sans"),
        hovertemplate="<b>%{x}</b><br>%{y:.2%}<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,30,54,0.5)",
        font=dict(family="DM Sans", color="#94a3b8"),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=13, color="#94a3b8"),
            linecolor="rgba(255,255,255,0.05)"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.04)",
            tickformat=".0%",
            range=[0, min(1.0, float(max(prediction)) * 1.25)],
            tickfont=dict(size=11, color="#475569"),
        ),
        margin=dict(l=10, r=10, t=20, b=10),
        height=280,
        bargap=0.35,
        hoverlabel=dict(
            bgcolor="#0f2340",
            bordercolor="#1e3a5f",
            font=dict(color="white", family="DM Sans")
        )
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

else:
    st.markdown("""
    <div style="text-align:center; padding: 2.5rem 1rem; color:#334155;">
        <div style="font-size:3rem; margin-bottom:0.8rem; opacity:0.4;">🧠</div>
        <p style="font-size:14px; font-weight:400;">Awaiting image upload…</p>
    </div>
    """, unsafe_allow_html=True)

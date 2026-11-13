
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import time

# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EcoSort AI | Waste Intelligence",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = "best_model.pth"

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

CLASS_INFO = {
    "cardboard": ("📦", "Cardboard", "Paper-based packaging"),
    "glass": ("🍾", "Glass", "Glass bottles and containers"),
    "metal": ("🥫", "Metal", "Metal cans and objects"),
    "paper": ("📄", "Paper", "Paper sheets and products"),
    "plastic": ("🧴", "Plastic", "Plastic bottles and packaging"),
    "trash": ("🗑️", "Trash", "General non-recyclable waste")
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #080d0b;
    --panel: #101916;
    --panel2: #131f1a;
    --green: #10b981;
    --green2: #34d399;
    --text: #f0fdf4;
    --muted: #94a3a0;
    --border: rgba(52,211,153,0.13);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1200px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}

#MainMenu, footer {
    visibility: hidden;
}

/* Navbar */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 22px;
    background: rgba(16,25,22,0.85);
    border: 1px solid var(--border);
    border-radius: 18px;
    margin-bottom: 45px;
}

.brand {
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -1px;
}

.brand span {
    color: var(--green2);
}

.nav-tag {
    color: var(--muted);
    font-size: 12px;
    border: 1px solid var(--border);
    border-radius: 30px;
    padding: 8px 14px;
}

/* Hero */
.hero {
    padding: 38px 0 55px;
    text-align: center;
}

.eyebrow {
    display: inline-block;
    color: #6ee7b7;
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 30px;
    padding: 9px 16px;
    font-size: 12px;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 24px;
}

.hero h1 {
    font-size: clamp(36px, 6vw, 72px);
    font-weight: 800;
    letter-spacing: -4px;
    line-height: 1.05;
    margin: 0;
}

.hero h1 span {
    color: #34d399;
}

.hero p {
    color: #94a3a0;
    font-size: 16px;
    line-height: 1.8;
    max-width: 650px;
    margin: 22px auto;
}

/* Cards */
.panel {
    background: linear-gradient(145deg, #111b16, #0c1410);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 25px;
    box-shadow: 0 12px 45px rgba(0,0,0,0.15);
}

.section-title {
    font-size: 25px;
    font-weight: 750;
    letter-spacing: -1px;
    margin-bottom: 8px;
}

.section-sub {
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 24px;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: #0c1510;
    border: 1px dashed rgba(52,211,153,0.35);
    border-radius: 18px;
    padding: 15px;
}

[data-testid="stFileUploader"] section {
    background: transparent;
}

[data-testid="stFileUploader"] button {
    background: #153c2b;
    color: #d1fae5;
    border: 1px solid #236447;
    border-radius: 10px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #10b981, #059669);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 22px;
    font-weight: 700;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    background: linear-gradient(135deg, #34d399, #10b981);
    color: #052e16;
    box-shadow: 0 8px 25px rgba(16,185,129,0.2);
}

/* Metrics */
.metric-card {
    background: #101a14;
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
}

.metric-value {
    font-size: 32px;
    font-weight: 800;
    color: #34d399;
}

.metric-label {
    color: #94a3a0;
    font-size: 12px;
    margin-top: 5px;
}

/* Category cards */
.category-card {
    background: #101916;
    border: 1px solid rgba(52,211,153,0.12);
    border-radius: 16px;
    padding: 17px 12px;
    text-align: center;
    transition: 0.2s ease;
    height: 100%;
}

.category-card:hover {
    border-color: #10b981;
    transform: translateY(-3px);
}

.category-icon {
    font-size: 30px;
    margin-bottom: 9px;
}

.category-name {
    font-size: 13px;
    font-weight: 700;
}

.category-desc {
    color: #82938a;
    font-size: 10px;
    margin-top: 5px;
}

.result-box {
    background: linear-gradient(145deg, #10251a, #0c1711);
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 22px;
    padding: 25px;
    margin-top: 20px;
}

.result-label {
    color: #6ee7b7;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.result-name {
    font-size: 34px;
    font-weight: 800;
    color: #ecfdf5;
    margin: 10px 0;
}

.confidence {
    font-size: 25px;
    font-weight: 800;
    color: #34d399;
}

.footer {
    text-align: center;
    padding: 35px 0 10px;
    color: #64746a;
    font-size: 12px;
    border-top: 1px solid rgba(52,211,153,0.1);
    margin-top: 45px;
}

.footer strong {
    color: #34d399;
}

a {
    color: #6ee7b7 !important;
}

hr {
    border-color: rgba(52,211,153,0.12);
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_model():

    model = models.mobilenet_v2(weights=None)

    model.classifier[1] = nn.Linear(
        model.last_channel,
        len(CLASSES)
    )

    if not os.path.exists(MODEL_PATH):
        return None

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get(
                "state_dict",
                checkpoint.get("model_state_dict", checkpoint)
            )
        else:
            state_dict = checkpoint

        # Handle DataParallel checkpoints
        state_dict = {
            key.replace("module.", ""): value
            for key, value in state_dict.items()
        }

        model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_image(model, image):

    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    predicted_index = probabilities.argmax().item()

    return (
        CLASSES[predicted_index],
        probabilities[predicted_index].item(),
        probabilities.cpu().tolist()
    )


# =========================================================
# NAVBAR
# =========================================================

st.markdown("""
<div class="navbar">
    <div class="brand">♻ Eco<span>Sort</span> AI</div>
    <div class="nav-tag">● AI WASTE INTELLIGENCE</div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">✦ COMPUTER VISION · SUSTAINABILITY</div>
    <h1>Waste Intelligence.<br><span>Powered by AI.</span></h1>
    <p>
        A smarter approach to waste identification.
        Upload an image, let artificial intelligence analyse it,
        and discover its predicted waste category.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# MODEL STATUS
# =========================================================

model = load_model()

if model is None:
    st.error(
        f"Model file '{MODEL_PATH}' was not found. "
        "Place your trained model in the same folder as app.py."
    )


# =========================================================
# CLASSIFICATION WORKSPACE
# =========================================================

st.markdown("""
<div class="panel">
    <div class="section-title">♻ Waste Classification Studio</div>
    <div class="section-sub">
        Upload a waste image to generate an AI prediction.
    </div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

with left:

    st.markdown("### 01 / Upload Image")

    uploaded_file = st.file_uploader(
        "Drop an image here or browse files",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a clear image of one waste object."
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True
        )

with right:

    st.markdown("### 02 / AI Analysis")

    if uploaded_file and model:

        st.markdown("""
        <div class="section-sub">
            Image ready for classification.
        </div>
        """, unsafe_allow_html=True)

        if st.button("✦ Analyse Waste Image", use_container_width=True):

            with st.spinner("AI is analysing your image..."):
                time.sleep(0.4)

                category, confidence, probabilities = predict_image(
                    model,
                    image
                )

            st.session_state["prediction"] = {
                "category": category,
                "confidence": confidence,
                "probabilities": probabilities
            }

    elif not uploaded_file:

        st.markdown("""
        <div class="result-box" style="text-align:center;">
            <div style="font-size:50px;">🌿</div>
            <h3 style="color:#d1fae5;">Ready to analyse</h3>
            <p style="color:#94a3a0;">
                Upload an image to begin.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# PREDICTION RESULTS
# =========================================================

if "prediction" in st.session_state:

    result = st.session_state["prediction"]

    category = result["category"]
    confidence = result["confidence"]
    probabilities = result["probabilities"]

    emoji, display_name, description = CLASS_INFO[category]

    st.markdown("---")
    st.markdown("## ✦ Classification Result")

    result_left, result_right = st.columns([1, 1])

    with result_left:

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">AI Predicted Category</div>
            <div style="font-size:55px;margin-top:15px;">{emoji}</div>
            <div class="result-name">{display_name}</div>
            <div style="color:#94a3a0;">{description}</div>
            <br>
            <div class="result-label">Model Confidence</div>
            <div class="confidence">{confidence * 100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            "Confidence is the model's estimated probability for "
            "its selected category, not a guarantee of correctness."
        )

    with result_right:

        st.markdown("### Category Probability")

        for i, class_name in enumerate(CLASSES):

            probability = probabilities[i] * 100

            st.write(
                f"**{CLASS_INFO[class_name][1]}** — {probability:.2f}%"
            )

            st.progress(probability / 100)


# =========================================================
# CATEGORY EXPLORER
# =========================================================

st.markdown("---")

st.markdown("""
<div class="section-title">The Six Waste Categories</div>
<div class="section-sub">
    EcoSort AI recognises six predefined waste classes.
</div>
""", unsafe_allow_html=True)

cols = st.columns(3)

for index, class_name in enumerate(CLASSES):

    emoji, name, description = CLASS_INFO[class_name]

    with cols[index % 3]:

        st.markdown(f"""
        <div class="category-card">
            <div class="category-icon">{emoji}</div>
            <div class="category-name">{name}</div>
            <div class="category-desc">{description}</div>
        </div>
        """, unsafe_allow_html=True)

    if index == 2:
        cols = st.columns(3)


# =========================================================
# ABOUT MODEL
# =========================================================

st.markdown("---")

st.markdown("""
<div class="section-title">Built with Intelligence</div>
<div class="section-sub">
    The technology behind EcoSort AI.
</div>
""", unsafe_allow_html=True)

tech1, tech2, tech3 = st.columns(3)

with tech1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">AI</div>
        <div class="metric-label">Computer Vision</div>
    </div>
    """, unsafe_allow_html=True)

with tech2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">CNN</div>
        <div class="metric-label">MobileNetV2 Architecture</div>
    </div>
    """, unsafe_allow_html=True)

with tech3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">84%</div>
        <div class="metric-label">Best Validation Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="panel" style="margin-top:20px;">
    <h3 style="color:#6ee7b7;">How it works</h3>
    <p style="color:#94a3a0;line-height:1.8;">
        EcoSort AI uses MobileNetV2 with transfer learning.
        The model processes an uploaded image, extracts visual
        features and predicts one of six waste categories.
        The trained model is integrated into a Streamlit interface
        for interactive classification.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# PROJECT LINKS
# =========================================================

st.markdown("---")

st.markdown("""
<div class="section-title">Explore the Project</div>
<div class="section-sub">
    Discover the code, dataset and development resources.
</div>
""", unsafe_allow_html=True)

link1, link2, link3 = st.columns(3)

with link1:
    st.markdown("""
    <div class="metric-card">
        <h3>⌘ GitHub</h3>
        <p class="metric-label">Source code & project files</p>
        <a href="https://github.com/" target="_blank">Open GitHub ↗</a>
    </div>
    """, unsafe_allow_html=True)

with link2:
    st.markdown("""
    <div class="metric-card">
        <h3>◈ Kaggle</h3>
        <p class="metric-label">Waste image dataset</p>
        <a href="https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset"
        target="_blank">Open Dataset ↗</a>
    </div>
    """, unsafe_allow_html=True)

with link3:
    st.markdown("""
    <div class="metric-card">
        <h3>◉ Streamlit</h3>
        <p class="metric-label">Application framework</p>
        <a href="https://streamlit.io/" target="_blank">Visit Streamlit ↗</a>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    <div style="font-size:22px;margin-bottom:10px;">♻</div>
    <strong>EcoSort AI</strong><br>
    Waste Intelligence. Powered by AI.<br><br>
    Developed by <strong>Siddharth Verma</strong> · Class 10 D<br>
    Artificial Intelligence · Computer Vision · Sustainability<br><br>
    <span style="color:#4b6355;">
        A student project demonstrating AI-based waste classification.
    </span>
</div>
""", unsafe_allow_html=True)

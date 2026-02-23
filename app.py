import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Set page config
st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Modern dark theme with glass-morphism ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: #0d0d14;
}

/* dark-panel behind sidebar */
[data-testid="stSidebar"] {
    background: #0a0a10 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

/* main container width / padding */
.main .block-container {
    max-width: 860px;
    padding: 2.5rem 2rem 3rem;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 1rem 0 0.5rem;
}
.app-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(139,92,246,0.12);
    border: 1px solid rgba(139,92,246,0.3);
    color: #c4b5fd;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 100px;
    margin-bottom: 1.4rem;
}
.app-title {
    font-size: clamp(2.4rem, 5vw, 3.6rem);
    font-weight: 900;
    letter-spacing: -2px;
    line-height: 1.05;
    background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 60%, #a1f0c0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
}
.app-subtitle {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.38);
    font-weight: 400;
    line-height: 1.65;
    max-width: 440px;
    margin: 0 auto 2rem;
}

/* ── Divider ── */
.gradient-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(139,92,246,0.35) 30%, rgba(96,165,250,0.35) 70%, transparent 100%);
    border: none;
    margin: 0.5rem 0 1.8rem;
}

/* ── Glass cards ── */
.glass-card {
    background: rgba(255,255,255,0.033);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1.8rem;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    margin-bottom: 1.5rem;
}
.card-section-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.28);
    margin-bottom: 1.1rem;
}

/* ── File uploader override ── */
[data-testid="stFileUploader"] > section {
    background: rgba(139,92,246,0.04) !important;
    border: 2px dashed rgba(139,92,246,0.35) !important;
    border-radius: 14px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"] > section:hover {
    border-color: rgba(139,92,246,0.7) !important;
    background: rgba(139,92,246,0.08) !important;
}
[data-testid="stFileUploader"] label {
    color: rgba(255,255,255,0.55) !important;
}

/* ── Image display card ── */
.image-display-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1rem;
    margin-bottom: 1.5rem;
}

/* ── Result cards ── */
.result-card {
    border-radius: 22px;
    padding: 2.2rem 2rem;
    text-align: center;
    margin: 1.6rem 0 1.2rem;
    position: relative;
    overflow: hidden;
}
.result-card::after {
    content: '';
    position: absolute;
    top: -60%;
    left: 50%;
    transform: translateX(-50%);
    width: 160%;
    height: 120%;
    border-radius: 50%;
    opacity: 0.07;
    filter: blur(40px);
    pointer-events: none;
}
.ai-result-card {
    background: linear-gradient(145deg, rgba(239,68,68,0.12) 0%, rgba(249,115,22,0.08) 100%);
    border: 1px solid rgba(239,68,68,0.25);
    box-shadow: 0 0 40px rgba(239,68,68,0.07), inset 0 1px 0 rgba(255,255,255,0.05);
}
.ai-result-card::after { background: #ef4444; }
.real-result-card {
    background: linear-gradient(145deg, rgba(16,185,129,0.12) 0%, rgba(59,130,246,0.08) 100%);
    border: 1px solid rgba(16,185,129,0.25);
    box-shadow: 0 0 40px rgba(16,185,129,0.07), inset 0 1px 0 rgba(255,255,255,0.05);
}
.real-result-card::after { background: #10b981; }

.result-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 100px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    padding: 4px 12px;
    margin-bottom: 1.1rem;
}
.ai-chip {
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.3);
    color: #fca5a5;
}
.real-chip {
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.3);
    color: #6ee7b7;
}
.result-main-icon { font-size: 2.6rem; display: block; margin-bottom: 0.4rem; }
.result-verdict {
    font-size: 1.65rem;
    font-weight: 800;
    color: #fff;
    letter-spacing: -0.8px;
    margin-bottom: 1rem;
}
.result-confidence-num {
    font-size: 4.5rem;
    font-weight: 900;
    letter-spacing: -3px;
    line-height: 1;
    margin-bottom: 0.15rem;
}
.ai-num  { color: #fb923c; }
.real-num { color: #34d399; }
.result-confidence-label {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.35);
    font-weight: 500;
    letter-spacing: 0.5px;
}

/* ── Probability breakdown ── */
.breakdown-card {
    background: rgba(255,255,255,0.028);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-top: 0.5rem;
}
.breakdown-row {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    margin-bottom: 1.1rem;
}
.breakdown-row:last-child { margin-bottom: 0; }
.breakdown-icon-wrap {
    width: 38px; height: 38px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.05rem; flex-shrink: 0;
}
.ai-icon-wrap  { background: rgba(239,68,68,0.12); }
.real-icon-wrap { background: rgba(16,185,129,0.12); }
.breakdown-info { flex: 1; }
.breakdown-name {
    font-size: 0.82rem;
    font-weight: 600;
    color: rgba(255,255,255,0.65);
    margin-bottom: 5px;
}
.track {
    height: 5px;
    background: rgba(255,255,255,0.07);
    border-radius: 6px;
    overflow: hidden;
}
.fill-ai   { height:100%; border-radius:6px; background: linear-gradient(90deg,#ef4444,#fb923c); }
.fill-real { height:100%; border-radius:6px; background: linear-gradient(90deg,#10b981,#34d399); }
.breakdown-pct {
    font-size: 0.9rem;
    font-weight: 700;
    color: rgba(255,255,255,0.85);
    min-width: 48px;
    text-align: right;
}

/* ── Info callout ── */
.info-callout {
    display: flex;
    gap: 0.8rem;
    align-items: flex-start;
    background: rgba(96,165,250,0.07);
    border: 1px solid rgba(96,165,250,0.18);
    border-radius: 12px;
    padding: 0.95rem 1.1rem;
    margin-top: 1.2rem;
}
.info-callout-icon { font-size: 1rem; line-height: 1.6; flex-shrink: 0; }
.info-callout-body { font-size: 0.82rem; color: rgba(255,255,255,0.48); line-height: 1.6; }
.info-callout-body strong { color: rgba(255,255,255,0.75); font-weight: 600; }

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 2rem 0 0.5rem;
    margin-top: 2rem;
    border-top: 1px solid rgba(255,255,255,0.05);
}
.footer-stack {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 10px;
    margin-bottom: 0.55rem;
    flex-wrap: wrap;
}
.footer-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 100px;
    padding: 3px 10px;
    font-size: 0.73rem;
    color: rgba(255,255,255,0.3);
    font-weight: 500;
}
.footer-disclaimer {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.2);
    line-height: 1.6;
}

/* ── Sidebar ── */
.sb-section { margin-bottom: 1.6rem; }
.sb-label {
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.25);
    margin-bottom: 0.8rem;
}
.sb-stat-card {
    background: rgba(255,255,255,0.038);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.5rem;
}
.sb-stat-value {
    font-size: 1.6rem; font-weight: 800; color: #fff; line-height: 1;
}
.sb-stat-label {
    font-size: 0.75rem; color: rgba(255,255,255,0.35); margin-top: 2px;
}
.sb-step {
    display: flex; gap: 0.7rem; align-items: flex-start; margin-bottom: 0.85rem;
}
.sb-step-num {
    width: 22px; height: 22px; border-radius: 7px;
    background: rgba(139,92,246,0.18); color: #c4b5fd;
    font-size: 0.7rem; font-weight: 800;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; margin-top: 1px;
}
.sb-step-text {
    font-size: 0.82rem; color: rgba(255,255,255,0.42); line-height: 1.55;
}
.sb-accent { color: #a78bfa; font-weight: 600; }
.sb-tag-row { display: flex; flex-wrap: wrap; gap: 6px; }
.sb-tag {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 6px;
    padding: 3px 9px;
    font-size: 0.73rem;
    color: rgba(255,255,255,0.38);
    font-weight: 500;
}

/* spinner color override */
.stSpinner > div > div { border-top-color: #8b5cf6 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1.5rem;">
        <div style="font-size:1.25rem; font-weight:800; color:#e0c3fc; letter-spacing:-0.5px; margin-bottom:0.3rem;">
            🔍 AI Detector
        </div>
        <div style="font-size:0.78rem; color:rgba(255,255,255,0.3); line-height:1.5;">
            Powered by ResNet50 &amp; PyTorch
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Model Performance</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-stat-card">
        <div class="sb-stat-value">85.68%</div>
        <div class="sb-stat-label">Test accuracy</div>
    </div>
    <div class="sb-stat-card">
        <div class="sb-stat-value">152K+</div>
        <div class="sb-stat-label">Training images</div>
    </div>
    <div class="sb-stat-card">
        <div class="sb-stat-value">ResNet50</div>
        <div class="sb-stat-label">Model architecture</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sb-label">How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-step">
        <div class="sb-step-num">1</div>
        <div class="sb-step-text">Upload a <span class="sb-accent">JPG or PNG</span> image using the uploader</div>
    </div>
    <div class="sb-step">
        <div class="sb-step-num">2</div>
        <div class="sb-step-text">The image is resized and <span class="sb-accent">normalized</span> for the model</div>
    </div>
    <div class="sb-step">
        <div class="sb-step-num">3</div>
        <div class="sb-step-text">ResNet50 runs <span class="sb-accent">inference</span> and produces class probabilities</div>
    </div>
    <div class="sb-step">
        <div class="sb-step-num">4</div>
        <div class="sb-step-text">You see a verdict with <span class="sb-accent">confidence score</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sb-label">Supported Formats</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-tag-row">
        <span class="sb-tag">JPG</span>
        <span class="sb-tag">JPEG</span>
        <span class="sb-tag">PNG</span>
    </div>
    """, unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# ── Transform ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-eyebrow">⚡ Deep Learning · Computer Vision</div>
    <div class="app-title">AI Image Detector</div>
    <div class="app-subtitle">
        Upload any image and our fine-tuned ResNet50 model will determine
        whether it was created by an AI or captured in reality.
    </div>
</div>
<hr class="gradient-divider">
""", unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────────────────
col_left, col_main, col_right = st.columns([0.5, 9, 0.5])

with col_main:
    # Upload section
    st.markdown('<div class="glass-card"><div class="card-section-label">Upload Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your image here, or click to browse",
        type=['jpg', 'jpeg', 'png'],
        help="Supported formats: JPG, JPEG, PNG",
        label_visibility="visible"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Display image in a styled card
        st.markdown('<div class="image-display-card">', unsafe_allow_html=True)
        img_col1, img_col2, img_col3 = st.columns([1, 6, 1])
        with img_col2:
            st.image(image, use_container_width=True)
        st.markdown(f"""
        <div style="text-align:center; margin-top:0.5rem;">
            <span style="font-size:0.75rem; color:rgba(255,255,255,0.3); font-weight:500;">
                {uploaded_file.name} &nbsp;·&nbsp; {image.size[0]}×{image.size[1]}px
            </span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Run inference
        with st.spinner("Analyzing image…"):
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)

        ai_prob   = probs[0].item() * 100
        real_prob = probs[1].item() * 100
        is_ai     = ai_prob > real_prob

        # ── Result card ──────────────────────────────────────────────────────
        if is_ai:
            st.markdown(f"""
            <div class="result-card ai-result-card">
                <div class="result-chip ai-chip">🤖 &nbsp;Detection Result</div>
                <div class="result-verdict">AI-Generated Image</div>
                <div class="result-confidence-num ai-num">{ai_prob:.1f}%</div>
                <div class="result-confidence-label">confidence score</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card real-result-card">
                <div class="result-chip real-chip">✅ &nbsp;Detection Result</div>
                <div class="result-verdict">Real / Authentic Image</div>
                <div class="result-confidence-num real-num">{real_prob:.1f}%</div>
                <div class="result-confidence-label">confidence score</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Probability breakdown ─────────────────────────────────────────────
        st.markdown(f"""
        <div class="breakdown-card">
            <div class="card-section-label">Probability Breakdown</div>
            <div class="breakdown-row">
                <div class="breakdown-icon-wrap ai-icon-wrap">🤖</div>
                <div class="breakdown-info">
                    <div class="breakdown-name">AI-Generated</div>
                    <div class="track">
                        <div class="fill-ai" style="width:{ai_prob:.1f}%"></div>
                    </div>
                </div>
                <div class="breakdown-pct">{ai_prob:.1f}%</div>
            </div>
            <div class="breakdown-row">
                <div class="breakdown-icon-wrap real-icon-wrap">📷</div>
                <div class="breakdown-info">
                    <div class="breakdown-name">Real / Authentic</div>
                    <div class="track">
                        <div class="fill-real" style="width:{real_prob:.1f}%"></div>
                    </div>
                </div>
                <div class="breakdown-pct">{real_prob:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Info callout ──────────────────────────────────────────────────────
        st.markdown("""
        <div class="info-callout">
            <div class="info-callout-icon">💡</div>
            <div class="info-callout-body">
                This model was trained on <strong>152,000+ images</strong> and achieves
                <strong>85.68% accuracy</strong> on held-out test data.
                Results are probabilistic — always apply human judgment for critical decisions.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <div class="footer-stack">
        <span class="footer-pill">🔥 PyTorch</span>
        <span class="footer-pill">🌊 Streamlit</span>
        <span class="footer-pill">🧠 ResNet50</span>
        <span class="footer-pill">🤗 HuggingFace Datasets</span>
    </div>
    <div class="footer-disclaimer">
        ⚠️ Demonstration project — results may not be 100% accurate. Not intended for forensic or legal use.
    </div>
</div>
""", unsafe_allow_html=True)

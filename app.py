import os
import ssl
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
ssl._create_default_https_context = ssl.create_default_context

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Neural Style Transfer",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Geist+Mono:wght@400;500&display=swap');

/* ── Tokens ── */
:root {
    --bg:        #000000;
    --surface:   #141417;
    --surface2:  #1c1c21;
    --border:    #2a2a30;
    --border2:   #333340;
    --primary:   #6366f1;
    --primary-h: #4f52d4;
    --text:      #f4f4f5;
    --text-2:    #a1a1aa;
    --text-3:    #71717a;
    --mono:      'Geist Mono', monospace;
    --sans:      'Plus Jakarta Sans', sans-serif;
    --r:         8px;
}

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--bg) !important;
    font-family: var(--sans) !important;
    color: var(--text) !important;
}

/* Hide chrome */
#MainMenu, footer, header,
[data-testid="stDecoration"],
[data-testid="stToolbar"],
[data-testid="stSidebar"],
[data-testid="collapsedControl"] { display: none !important; }

/* Container width */
[data-testid="block-container"] {
    max-width: 680px !important;
    padding: 3rem 1.25rem 5rem !important;
    margin: 0 auto !important;
}

/* All text */
p, span, div, label, li, td, th, a,
[data-testid="stMarkdownContainer"] p {
    font-family: var(--sans) !important;
}

/* ── HEADER ── */
.app-eyebrow {
    font-family: var(--mono) !important;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--primary);
    margin-bottom: 0.6rem;
    display: block;
}
.app-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.03em;
    margin: 0 0 0.35rem;
    line-height: 1.2;
}
.app-sub {
    font-size: 0.85rem;
    color: var(--text-2);
    margin: 0 0 1.75rem;
    line-height: 1.65;
    font-weight: 400;
}
.app-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

/* ── DEVICE BADGE ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 100px;
    font-family: var(--mono) !important;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
    font-weight: 500;
}
.badge-cpu {
    background: rgba(99,102,241,0.12);
    color: #818cf8;
    border: 1px solid rgba(99,102,241,0.25);
}
.badge-gpu {
    background: rgba(34,197,94,0.1);
    color: #4ade80;
    border: 1px solid rgba(34,197,94,0.2);
}
.badge-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: currentColor;
    display: inline-block;
}

/* ── SECTION LABEL ── */
.sec-label {
    font-family: var(--mono) !important;
    font-size: 0.58rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-3);
    margin-bottom: 0.65rem;
    display: block;
}

/* ── HINT TEXT ── */
.hint {
    font-size: 0.78rem;
    color: var(--text-3);
    margin-top: 0.4rem;
    line-height: 1.55;
}

/* ── STATUS TEXT ── */
.status {
    font-family: var(--mono) !important;
    font-size: 0.72rem;
    color: var(--text-3);
}

/* ════════════════════════════════
   WIDGET OVERRIDES
   ════════════════════════════════ */

/* Labels */
[data-testid="stWidgetLabel"] p,
.stSlider label, .stSelectbox label,
.stRadio label, .stFileUploader label {
    font-family: var(--sans) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: var(--text-2) !important;
    letter-spacing: 0;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    width: 100%;
}
[data-testid="stFileUploader"] > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    padding: 1.25rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: var(--primary) !important;
}
/* The "Browse files" button inside uploader */
[data-testid="stFileUploaderDropzone"] button {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 6px !important;
    font-family: var(--sans) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 0.35rem 0.8rem !important;
    white-space: nowrap !important;
    box-shadow: none !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    border-color: var(--primary) !important;
}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {
    font-family: var(--sans) !important;
    font-size: 0.75rem !important;
    color: var(--text-3) !important;
}
/* Label above uploader */
[data-testid="stFileUploader"] label {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: var(--text-2) !important;
    margin-bottom: 0.4rem !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
    font-size: 0.85rem !important;
    min-height: 40px !important;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: var(--border2) !important;
}
[data-baseweb="select"] svg { color: var(--text-3) !important; }
[data-baseweb="popover"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
}
[data-baseweb="menu"] li {
    font-family: var(--sans) !important;
    font-size: 0.83rem !important;
    color: var(--text) !important;
}
[data-baseweb="menu"] li:hover { background: var(--border) !important; }

/* ── Sliders — kill all red ── */
[data-baseweb="slider"] {
    padding-top: 0.25rem !important;
}
/* Track background */
[data-baseweb="slider"] > div > div > div {
    background: var(--border) !important;
    border-radius: 99px !important;
    height: 3px !important;
}
/* Filled portion */
[data-baseweb="slider"] > div > div > div:first-child {
    background: var(--primary) !important;
}
/* Thumb */
[data-baseweb="slider"] [role="slider"] {
    background: var(--primary) !important;
    border: 2px solid var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.2) !important;
    width: 14px !important;
    height: 14px !important;
}
[data-baseweb="slider"] [role="slider"]:hover {
    box-shadow: 0 0 0 5px rgba(99,102,241,0.25) !important;
}
div[class*="StyledThumb"] {
    background: var(--primary) !important;
    border-color: var(--primary) !important;
}
div[class*="StyledInnerTrack"]:first-child {
    background: var(--primary) !important;
}
/* Tooltip on slider */
[data-baseweb="tooltip"] div {
    background: var(--surface2) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.65rem !important;
    border: 1px solid var(--border2) !important;
}

/* ── Radio — pill style, no red dot ── */
[data-testid="stRadio"] > div {
    gap: 0.5rem !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    margin-top: 0.35rem !important;
}
[data-testid="stRadio"] label {
    display: flex !important;
    align-items: center !important;
    gap: 0 !important;
    padding: 0.35rem 1rem !important;
    border-radius: 100px !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    font-family: var(--sans) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: var(--text-2) !important;
}
[data-testid="stRadio"] label:has(input:checked) {
    border-color: var(--primary) !important;
    background: rgba(99,102,241,0.12) !important;
    color: #c7d2fe !important;
}
[data-testid="stRadio"] label:hover {
    border-color: var(--border2) !important;
    color: var(--text) !important;
}
/* Hide the actual radio circle */
[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {
    display: none !important;
}

/* ── Primary button ── */
[data-testid="baseButton-primary"] {
    background: var(--primary) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--r) !important;
    font-family: var(--sans) !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    letter-spacing: 0 !important;
    height: 42px !important;
    transition: background 0.15s !important;
    box-shadow: none !important;
}
[data-testid="baseButton-primary"]:hover {
    background: var(--primary-h) !important;
}
[data-testid="baseButton-primary"]:focus,
[data-testid="baseButton-primary"]:active {
    background: var(--primary-h) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.3) !important;
}

/* ── Download / secondary button ── */
[data-testid="baseButton-secondary"] {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    font-family: var(--sans) !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    height: 42px !important;
    box-shadow: none !important;
}
[data-testid="baseButton-secondary"]:hover {
    border-color: var(--primary) !important;
}

/* ── Alerts ── */
[data-testid="stAlert"],
[data-baseweb="notification"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-left: 2px solid var(--primary) !important;
    border-radius: var(--r) !important;
    font-family: var(--sans) !important;
    font-size: 0.83rem !important;
    color: var(--text) !important;
}
[data-testid="stAlert"] svg,
[data-baseweb="notification"] svg {
    fill: var(--primary) !important;
    color: var(--primary) !important;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div {
    background: var(--surface2) !important;
    border-radius: 99px !important;
    height: 3px !important;
}
[data-testid="stProgress"] > div > div {
    background: var(--primary) !important;
    border-radius: 99px !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--sans) !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    color: var(--text-2) !important;
    padding: 0.85rem 1rem !important;
}
[data-testid="stExpander"] summary:hover { color: var(--text) !important; }
[data-testid="stExpander"] > div > div { padding: 0 1rem 1rem !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] p {
    font-family: var(--sans) !important;
    font-size: 0.83rem !important;
    color: var(--text-3) !important;
}

/* ── Images ── */
[data-testid="stImage"] img {
    border-radius: var(--r) !important;
}
[data-testid="caption"] {
    font-family: var(--mono) !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-3) !important;
    text-align: center !important;
    margin-top: 0.4rem !important;
}

/* ── Focus rings ── */
*:focus-visible {
    outline: 2px solid var(--primary) !important;
    outline-offset: 2px !important;
    box-shadow: none !important;
}

/* Kill any remaining inline red */
[style*="background-color: rgb(255, 75"],
[style*="background: rgb(255, 75"] {
    background-color: var(--primary) !important;
    background: var(--primary) !important;
}

/* ── Theory block ── */
.theory h4 {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text);
    margin: 0 0 0.75rem;
}
.theory p, .theory li {
    font-size: 0.82rem;
    color: var(--text-2);
    line-height: 1.75;
}
.theory li { margin-bottom: 0.35rem; }
.theory strong { color: var(--text); font-weight: 600; }
.theory code {
    font-family: var(--mono) !important;
    font-size: 0.72rem;
    background: rgba(99,102,241,0.1);
    color: #a5b4fc;
    padding: 1px 5px;
    border-radius: 4px;
}
.theory table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
.theory th {
    font-family: var(--mono) !important;
    font-size: 0.58rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-3);
    padding: 0.4rem 0.6rem;
    border-bottom: 1px solid var(--border);
    text-align: left;
    font-weight: 400;
}
.theory td {
    font-size: 0.8rem;
    padding: 0.45rem 0.6rem;
    border-bottom: 1px solid var(--border);
    color: var(--text-2);
}

/* ── Footer ── */
.app-footer {
    margin-top: 3rem;
    padding-top: 1.25rem;
    border-top: 1px solid var(--border);
    font-family: var(--mono) !important;
    font-size: 0.58rem;
    letter-spacing: 0.12em;
    color: var(--text-3);
    text-align: center;
    text-transform: uppercase;
}

[data-testid="stHorizontalBlock"] { gap: 0.75rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<span class="app-eyebrow">LLM &amp; Gen AI Lab &mdash; Experiment 6 &mdash; Task 2</span>
<h1 class="app-title">Neural Style Transfer</h1>
<p class="app-sub">Reimagine any photograph through the visual language of artistic masterworks, powered by VGG19.</p>
""", unsafe_allow_html=True)

if device.type == "cuda":
    st.markdown('<span class="badge badge-gpu"><span class="badge-dot"></span>&nbsp;GPU &mdash; Accelerated</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="badge badge-cpu"><span class="badge-dot"></span>&nbsp;CPU &mdash; May be slow</span>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def load_image(image_file, size):
    image = Image.open(image_file).convert("RGB")
    loader = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    return loader(image).unsqueeze(0).to(device, torch.float)

def tensor_to_pil(tensor):
    return transforms.ToPILImage()(tensor.cpu().clone().squeeze(0).clamp(0, 1))

class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
        self.loss = 0
    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

def gram_matrix(x):
    b, c, h, w = x.size()
    f = x.view(b * c, h * w)
    return torch.mm(f, f.t()).div(b * c * h * w)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = 0
    def forward(self, x):
        self.loss = nn.functional.mse_loss(gram_matrix(x), self.target)
        return x

class Normalization(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(-1, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).to(device).view(-1, 1, 1)
    def forward(self, x):
        return (x - self.mean) / self.std

def build_model_and_losses(cnn, style_img, content_img):
    content_layers = ['conv_4']
    style_layers   = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    model = nn.Sequential(Normalization().to(device))
    content_losses, style_losses = [], []
    conv_i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            conv_i += 1; name = f'conv_{conv_i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{conv_i}'; layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{conv_i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{conv_i}'
        else:
            continue
        model.add_module(name, layer)
        if name in content_layers:
            cl = ContentLoss(model(content_img).detach())
            model.add_module(f"content_loss_{conv_i}", cl)
            content_losses.append(cl)
        if name in style_layers:
            sl = StyleLoss(model(style_img).detach())
            model.add_module(f"style_loss_{conv_i}", sl)
            style_losses.append(sl)
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            break
    return model[:(i + 1)], style_losses, content_losses

@st.cache_resource(show_spinner=False)
def load_vgg():
    import requests, tempfile
    weights_url = "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"
    cache_path  = os.path.join(tempfile.gettempdir(), "vgg19_weights.pth")
    if not os.path.exists(cache_path):
        resp = requests.get(weights_url, stream=True, verify=certifi.where())
        resp.raise_for_status()
        with open(cache_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
    cnn = models.vgg19(weights=None)
    cnn.load_state_dict(torch.load(cache_path, map_location=device))
    return cnn.features.to(device).eval()

def run_style_transfer(cnn, content_img, style_img, num_steps, style_weight,
                       content_weight, progress_bar, status_placeholder):
    model, style_losses, content_losses = build_model_and_losses(cnn, style_img, content_img)
    input_img = content_img.clone().requires_grad_(True)
    model.requires_grad_(False)
    optimizer = optim.LBFGS([input_img])
    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad(): input_img.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            loss = style_weight   * sum(sl.loss for sl in style_losses) + \
                   content_weight * sum(cl.loss for cl in content_losses)
            loss.backward()
            run[0] += 1
            if run[0] % 10 == 0:
                pct = min(int(run[0] / num_steps * 100), 100)
                progress_bar.progress(pct)
                status_placeholder.markdown(
                    f'<p class="status">Step {run[0]}&thinsp;/&thinsp;{num_steps}&nbsp;&mdash;&nbsp;Loss: {loss.item():.4f}</p>',
                    unsafe_allow_html=True
                )
            return loss
        optimizer.step(closure)
    with torch.no_grad(): input_img.clamp_(0, 1)
    return input_img

# ─────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────
st.markdown('<hr class="app-divider">', unsafe_allow_html=True)
st.markdown('<span class="sec-label">Parameters</span>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    image_size = st.selectbox("Resolution", options=[128, 256, 512], index=0,
                               help="Smaller = faster. 128px recommended on CPU.")
    content_weight = st.slider("Content Weight", min_value=1, max_value=10, value=1)
with c2:
    num_steps = st.slider("Optimization Steps", min_value=100, max_value=600, value=200, step=50)
    style_weight = st.select_slider(
        "Style Weight",
        options=[1e4, 1e5, 5e5, 1e6],
        value=1e5,
        format_func=lambda x: f"{x:.0e}"
    )

# ─────────────────────────────────────────────
# CONTENT IMAGE
# ─────────────────────────────────────────────
st.markdown('<hr class="app-divider">', unsafe_allow_html=True)
st.markdown('<span class="sec-label">01 — Content Image</span>', unsafe_allow_html=True)

content_file = st.file_uploader(
    "Drop your content image here",
    type=["jpg", "jpeg", "png"],
    key="content"
)
if not content_file:
    st.markdown('<p class="hint">The photograph whose structure and composition will be preserved.</p>', unsafe_allow_html=True)
else:
    st.image(content_file, use_container_width=True)

# ─────────────────────────────────────────────
# STYLE REFERENCE
# ─────────────────────────────────────────────
st.markdown('<hr class="app-divider">', unsafe_allow_html=True)
st.markdown('<span class="sec-label">02 — Style Reference</span>', unsafe_allow_html=True)

PRESETS = {
    "Van Gogh — Starry Night":  "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
    "Picasso — Weeping Woman":  "https://upload.wikimedia.org/wikipedia/en/1/14/Picasso_The_Weeping_Woman_Tate_identifier_T05010_10.jpg",
    "Monet — Water Lilies":     "https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/Claude_Monet_-_Water_Lilies_-_1906%2C_Ryerson.jpg/1280px-Claude_Monet_-_Water_Lilies_-_1906%2C_Ryerson.jpg",
}

style_mode = st.radio(
    "Style source",
    ["Upload your own", "Use a preset"],
    horizontal=True,
    label_visibility="collapsed"
)

style_file      = None
selected_preset = None

if style_mode == "Upload your own":
    style_file = st.file_uploader(
        "Drop your style image here",
        type=["jpg", "jpeg", "png"],
        key="style"
    )
    if not style_file:
        st.markdown('<p class="hint">Any artwork or texture — its visual character will be transferred onto your content image.</p>', unsafe_allow_html=True)
    else:
        st.image(style_file, use_container_width=True)
else:
    selected_preset = st.selectbox("Choose a preset", list(PRESETS.keys()), label_visibility="collapsed")
    st.image(PRESETS[selected_preset], use_container_width=True)

# ─────────────────────────────────────────────
# GENERATE
# ─────────────────────────────────────────────
st.markdown('<hr class="app-divider">', unsafe_allow_html=True)
run_btn = st.button("Generate Stylized Image", type="primary", use_container_width=True)

if run_btn:
    if not content_file:
        st.warning("Please upload a content image first.")
    elif style_mode == "Upload your own" and not style_file:
        st.warning("Please upload a style image first.")
    else:
        try:
            with st.spinner("Loading VGG19 weights…"):
                cnn = load_vgg()

            content_img = load_image(content_file, image_size)

            if style_mode == "Upload your own":
                style_img = load_image(style_file, image_size)
            else:
                import requests
                resp = requests.get(PRESETS[selected_preset], headers={"User-Agent": "Mozilla/5.0"}, verify=certifi.where())
                resp.raise_for_status()
                style_img = load_image(io.BytesIO(resp.content), image_size)

            st.info(f"Running {num_steps} steps on {device.type.upper()} at {image_size}px — this may take a few minutes on CPU.")

            prog  = st.progress(0)
            sstxt = st.empty()

            output = run_style_transfer(cnn, content_img, style_img,
                                        num_steps, style_weight, content_weight,
                                        prog, sstxt)
            prog.progress(100)
            sstxt.success("Done! Your stylized image is ready.")
            result_pil = tensor_to_pil(output)

            st.markdown('<hr class="app-divider">', unsafe_allow_html=True)
            st.markdown('<span class="sec-label">Result</span>', unsafe_allow_html=True)

            r1, r2, r3 = st.columns(3)
            with r1:
                content_file.seek(0)
                st.image(content_file, caption="Content", use_container_width=True)
            with r2:
                if style_mode == "Upload your own":
                    style_file.seek(0)
                    st.image(style_file, caption="Style", use_container_width=True)
                else:
                    st.image(PRESETS[selected_preset], caption="Style", use_container_width=True)
            with r3:
                st.image(result_pil, caption="Output", use_container_width=True)

            buf = io.BytesIO()
            result_pil.save(buf, format="PNG")
            buf.seek(0)
            st.download_button("Download Stylized Image", data=buf,
                               file_name="stylized_output.png", mime="image/png",
                               use_container_width=True)

        except Exception as e:
            st.warning(f"Something went wrong: {e}")
            st.exception(e)

# ─────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────
st.markdown('<hr class="app-divider">', unsafe_allow_html=True)

with st.expander("How Neural Style Transfer works"):
    st.markdown("""
<div class="theory">
<h4>Technical Overview</h4>
<p>Neural Style Transfer uses a pretrained VGG19 CNN to extract <em>content</em> and <em>style</em>
representations from two images, then synthesizes a new image that combines both.</p>
<ul>
<li><strong>Content Loss</strong> — MSE between deep feature maps at <code>conv_4</code>, preserving structure.</li>
<li><strong>Style Loss</strong> — Gram matrix differences across five layers capture texture, color and brushwork.</li>
<li><strong>Gram Matrix</strong> — Reshape <code>(C,H,W)</code> to <code>(C, H&times;W)</code>, compute <code>G = F&middot;F&sup1;</code>.</li>
<li><strong>L-BFGS</strong> — Minimizes <code>style_weight &times; style_loss + content_weight &times; content_loss</code>.</li>
</ul>
<table>
<tr><th>Purpose</th><th>VGG19 layers</th></tr>
<tr><td>Content</td><td><code>conv_4</code></td></tr>
<tr><td>Style</td><td><code>conv_1, conv_2, conv_3, conv_4, conv_5</code></td></tr>
</table>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(
    '<p class="app-footer">Experiment 06 &middot; LLM &amp; Generative AI Lab &middot; '
    'Fr. Conceicao Rodrigues College of Engineering &middot; Dept. of Computer Engineering</p>',
    unsafe_allow_html=True
)
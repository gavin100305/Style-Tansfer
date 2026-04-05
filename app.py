import os
import ssl
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
ssl._create_default_https_context = ssl.create_default_context

import urllib.request
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
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CSS — dark theme from provided design tokens
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600&family=Geist+Mono:wght@400;500&family=Noto+Serif+Georgian:wght@400;600&display=swap');

/* ── Design tokens (dark only) ── */
:root {
    --background:           oklch(0 0 0);
    --foreground:           oklch(1 0 0);
    --card:                 oklch(0.2103 0 267.51);
    --card-foreground:      oklch(0.9461 0 0);
    --primary:              oklch(0.5144 0.1605 267.44);
    --primary-foreground:   oklch(0.97 0.014 254.604);
    --secondary:            oklch(0.25 0 0);
    --secondary-foreground: oklch(0.94 0 0);
    --muted:                oklch(0.23 0 0);
    --muted-foreground:     oklch(0.72 0 0);
    --accent:               oklch(0.32 0 0);
    --accent-foreground:    oklch(0.9214 0.0248 257.65);
    --border:               oklch(0.26 0 0);
    --input:                oklch(0.32 0 0);
    --ring:                 oklch(0.5144 0.1605 267.44);
    --destructive:          oklch(0.704 0.191 22.216);
    --success:              oklch(0.65 0.15 145);
    --radius:               0.5rem;
    --font-sans:            'Geist', sans-serif;
    --font-mono:            'Geist Mono', monospace;
    --font-serif:           'Noto Serif Georgian', ui-serif, serif;
}

/* ── Global reset ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"] {
    background-color: var(--background) !important;
    color: var(--foreground);
    font-family: var(--font-sans);
}

/* Hide all Streamlit chrome */
#MainMenu, footer, header,
[data-testid="stDecoration"],
[data-testid="stSidebarNav"],
[data-testid="collapsedControl"] { display: none !important; }

/* Sidebar fully hidden */
[data-testid="stSidebar"] { display: none !important; }

/* Main content max-width + padding */
[data-testid="block-container"] {
    max-width: 1100px !important;
    padding: 2.5rem 2rem 4rem 2rem !important;
    margin: 0 auto !important;
}

/* ── Typography ── */
h1, h2, h3 {
    font-family: var(--font-serif) !important;
    color: var(--foreground) !important;
    letter-spacing: -0.02em;
}

/* ── Page header ── */
.nst-header {
    padding: 2.5rem 0 2rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
}
.nst-header h1 {
    font-family: var(--font-serif);
    font-size: 2.4rem;
    font-weight: 600;
    color: var(--foreground);
    letter-spacing: -0.025em;
    margin: 0 0 0.5rem 0;
    line-height: 1.1;
}
.nst-header p {
    font-family: var(--font-sans);
    font-size: 0.9rem;
    font-weight: 300;
    color: var(--muted-foreground);
    margin: 0;
    line-height: 1.6;
}

/* ── Section label ── */
.section-label {
    display: block;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--primary);
    margin-bottom: 0.6rem;
}

/* ── Parameter card (inline, no sidebar) ── */
.param-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    margin-bottom: 2rem;
}
.param-cell {
    background: var(--card);
    padding: 1rem 1.25rem;
}
.param-cell-label {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted-foreground);
    margin-bottom: 0.3rem;
}
.param-cell-value {
    font-family: var(--font-sans);
    font-size: 0.92rem;
    font-weight: 500;
    color: var(--card-foreground);
}

/* ── Device badge ── */
.device-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.75rem;
    border-radius: calc(var(--radius) - 2px);
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}
.device-gpu { background: oklch(0.18 0.05 145); color: oklch(0.75 0.15 145); border: 1px solid oklch(0.28 0.08 145); }
.device-cpu { background: oklch(0.2 0.04 267); color: oklch(0.75 0.1 267);  border: 1px solid oklch(0.32 0.08 267); }

/* ── Upload zone hint ── */
.upload-hint {
    font-family: var(--font-sans);
    font-size: 0.78rem;
    color: var(--muted-foreground);
    margin-top: 0.35rem;
    line-height: 1.5;
}

/* ── Divider ── */
.nst-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
}

/* ── Status text ── */
.status-text {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--muted-foreground);
    letter-spacing: 0.02em;
}

/* ── Theory block ── */
.theory-block {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.75rem;
}
.theory-block h4 {
    font-family: var(--font-serif);
    font-size: 1.05rem;
    font-weight: 400;
    color: var(--card-foreground);
    margin: 0 0 1rem 0;
    letter-spacing: -0.01em;
}
.theory-block p, .theory-block li {
    font-family: var(--font-sans);
    font-size: 0.85rem;
    color: var(--muted-foreground);
    line-height: 1.75;
}
.theory-block li { margin-bottom: 0.4rem; }
.theory-block strong { color: var(--card-foreground); font-weight: 500; }
.theory-block code {
    font-family: var(--font-mono);
    font-size: 0.76rem;
    background: var(--muted);
    color: var(--accent-foreground);
    padding: 1px 6px;
    border-radius: 3px;
}
.theory-block table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 1.25rem;
    font-size: 0.82rem;
}
.theory-block th {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted-foreground);
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--border);
    text-align: left;
    font-weight: 400;
}
.theory-block td {
    font-family: var(--font-sans);
    padding: 0.55rem 0.75rem;
    border-bottom: 1px solid var(--border);
    color: var(--card-foreground);
}

/* ── Footer ── */
.nst-footer {
    margin-top: 3.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    color: var(--muted-foreground);
    text-align: center;
    text-transform: uppercase;
}

/* ── Streamlit widget overrides ── */

/* File uploader */
[data-testid="stFileUploader"] > div {
    background: var(--card) !important;
    border: 1px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    transition: border-color 0.15s ease;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: var(--primary) !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span {
    font-family: var(--font-sans) !important;
    color: var(--muted-foreground) !important;
    font-size: 0.83rem !important;
}

/* Radio buttons */
[data-testid="stRadio"] > div {
    gap: 0.5rem !important;
    flex-direction: row !important;
}
[data-testid="stRadio"] label {
    font-family: var(--font-sans) !important;
    font-size: 0.83rem !important;
    color: var(--muted-foreground) !important;
}
[data-testid="stRadio"] label:has(input:checked) {
    color: var(--foreground) !important;
}

/* Selectbox */
[data-testid="stSelectbox"] label {
    font-family: var(--font-sans) !important;
    font-size: 0.83rem !important;
    color: var(--muted-foreground) !important;
}
[data-testid="stSelectbox"] > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--card-foreground) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.85rem !important;
}

/* Sliders */
[data-testid="stSlider"] label,
[data-testid="stSelectSlider"] label {
    font-family: var(--font-sans) !important;
    font-size: 0.83rem !important;
    color: var(--muted-foreground) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"],
[data-testid="stSelectSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: var(--primary) !important;
    border-color: var(--primary) !important;
}

/* Primary button */
[data-testid="baseButton-primary"] {
    background: var(--primary) !important;
    color: var(--primary-foreground) !important;
    border: none !important;
    font-family: var(--font-sans) !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.02em !important;
    border-radius: var(--radius) !important;
    transition: opacity 0.15s ease !important;
}
[data-testid="baseButton-primary"]:hover { opacity: 0.88 !important; }

/* Secondary / download button */
[data-testid="baseButton-secondary"] {
    background: var(--card) !important;
    color: var(--card-foreground) !important;
    border: 1px solid var(--border) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.875rem !important;
    border-radius: var(--radius) !important;
    transition: border-color 0.15s ease !important;
}
[data-testid="baseButton-secondary"]:hover {
    border-color: var(--primary) !important;
    color: var(--foreground) !important;
}

/* Alerts */
[data-testid="stAlert"] {
    background: var(--card) !important;
    border-radius: var(--radius) !important;
    border-left-width: 2px !important;
    font-family: var(--font-sans) !important;
    font-size: 0.84rem !important;
    color: var(--card-foreground) !important;
}

/* Progress bar */
[data-testid="stProgress"] > div > div {
    background: var(--primary) !important;
    border-radius: 99px !important;
}
[data-testid="stProgress"] > div {
    background: var(--muted) !important;
    border-radius: 99px !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--font-sans) !important;
    font-size: 0.85rem !important;
    color: var(--muted-foreground) !important;
    padding: 0.85rem 1rem !important;
}
[data-testid="stExpander"] summary:hover {
    color: var(--foreground) !important;
}

/* Image captions */
[data-testid="caption"] {
    font-family: var(--font-mono) !important;
    font-size: 0.64rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted-foreground) !important;
    text-align: center !important;
}

/* Spinner */
[data-testid="stSpinner"] p {
    font-family: var(--font-sans) !important;
    font-size: 0.85rem !important;
    color: var(--muted-foreground) !important;
}

/* Column gap override */
[data-testid="stHorizontalBlock"] {
    gap: 1.5rem !important;
}
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
<div class="nst-header">
    <h1>Neural Style Transfer</h1>
    <p>Reimagine any photograph through the visual language of artistic masterworks, powered by VGG19.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPER FUNCTIONS — unchanged
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
                       content_weight, progress_bar, status_text):
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
                progress_bar.progress(min(int(run[0] / num_steps * 100), 100))
                status_text.markdown(
                    f'<p class="status-text">Step {run[0]} / {num_steps}&ensp;&mdash;&ensp;Loss: {loss.item():.4f}</p>',
                    unsafe_allow_html=True
                )
            return loss
        optimizer.step(closure)
    with torch.no_grad(): input_img.clamp_(0, 1)
    return input_img

# ─────────────────────────────────────────────
# PARAMETERS — inline, no sidebar
# ─────────────────────────────────────────────
if device.type == "cuda":
    st.markdown('<span class="device-badge device-gpu">GPU &mdash; Accelerated</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="device-badge device-cpu">CPU &mdash; May be slow</span>', unsafe_allow_html=True)

st.markdown('<span class="section-label">Parameters</span>', unsafe_allow_html=True)

p1, p2, p3, p4 = st.columns(4, gap="small")
with p1:
    image_size = st.selectbox(
        "Resolution",
        options=[128, 256, 512],
        index=0,
        help="Smaller = faster. 128px recommended on CPU."
    )
with p2:
    num_steps = st.slider("Optimization Steps", min_value=100, max_value=600, value=200, step=50)
with p3:
    style_weight = st.select_slider(
        "Style Weight",
        options=[1e4, 1e5, 5e5, 1e6],
        value=1e5,
        format_func=lambda x: f"{x:.0e}"
    )
with p4:
    content_weight = st.slider("Content Weight", min_value=1, max_value=10, value=1)

st.markdown('<hr class="nst-divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# IMAGE INPUTS
# ─────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<span class="section-label">01 &mdash; Content Image</span>', unsafe_allow_html=True)
    content_file = st.file_uploader(
        "content_upload",
        type=["jpg", "jpeg", "png"],
        key="content",
        label_visibility="collapsed"
    )
    st.markdown(
        '<p class="upload-hint">The photograph whose structure and composition will be preserved.</p>',
        unsafe_allow_html=True
    )
    if content_file:
        st.image(content_file, caption="Content", use_container_width=True)

with col2:
    st.markdown('<span class="section-label">02 &mdash; Style Reference</span>', unsafe_allow_html=True)

    preset_styles = {
        "Van Gogh — Starry Night": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
        "Picasso — Weeping Woman": "https://upload.wikimedia.org/wikipedia/en/1/14/Picasso_The_Weeping_Woman_Tate_identifier_T05010_10.jpg",
        "Monet — Water Lilies":    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/Claude_Monet_-_Water_Lilies_-_1906%2C_Ryerson.jpg/1280px-Claude_Monet_-_Water_Lilies_-_1906%2C_Ryerson.jpg",
    }

    style_option = st.radio(
        "style_source",
        ["Upload your own", "Use preset artwork"],
        horizontal=True,
        label_visibility="collapsed"
    )

    style_file     = None
    selected_preset = None

    if style_option == "Upload your own":
        style_file = st.file_uploader(
            "style_upload",
            type=["jpg", "jpeg", "png"],
            key="style",
            label_visibility="collapsed"
        )
        st.markdown(
            '<p class="upload-hint">Any artwork or texture — its visual character will be transferred onto the content image.</p>',
            unsafe_allow_html=True
        )
        if style_file:
            st.image(style_file, caption="Style Reference", use_container_width=True)
    else:
        selected_preset = st.selectbox(
            "preset_select",
            list(preset_styles.keys()),
            label_visibility="collapsed"
        )
        st.image(preset_styles[selected_preset], caption=selected_preset, use_container_width=True)

# ─────────────────────────────────────────────
# GENERATE BUTTON
# ─────────────────────────────────────────────
st.markdown('<hr class="nst-divider">', unsafe_allow_html=True)

generate = st.button("Generate Stylized Image", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# GENERATION LOGIC
# ─────────────────────────────────────────────
if generate:
    if not content_file:
        st.error("Please upload a content image before generating.")
    elif style_option == "Upload your own" and not style_file:
        st.error("Please upload a style reference image before generating.")
    else:
        try:
            with st.spinner("Loading VGG19 model weights — cached after the first run..."):
                cnn = load_vgg()

            content_img = load_image(content_file, image_size)

            if style_option == "Upload your own":
                style_img = load_image(style_file, image_size)
            else:
                import requests
                resp = requests.get(
                    preset_styles[selected_preset],
                    headers={"User-Agent": "Mozilla/5.0"},
                    verify=certifi.where()
                )
                resp.raise_for_status()
                style_img = load_image(io.BytesIO(resp.content), image_size)

            st.info(
                f"Running {num_steps} optimization steps on {device.type.upper()} "
                f"at {image_size}px. This may take several minutes on CPU."
            )

            progress_bar = st.progress(0)
            status_text  = st.empty()

            output = run_style_transfer(
                cnn, content_img, style_img,
                num_steps, style_weight, content_weight,
                progress_bar, status_text
            )

            progress_bar.progress(100)
            status_text.success("Transfer complete.")
            result_pil = tensor_to_pil(output)

            st.markdown('<hr class="nst-divider">', unsafe_allow_html=True)
            st.markdown('<span class="section-label">Result</span>', unsafe_allow_html=True)

            r1, r2, r3 = st.columns(3, gap="medium")
            with r1:
                content_file.seek(0)
                st.image(content_file, caption="Content", use_container_width=True)
            with r2:
                if style_option == "Upload your own":
                    style_file.seek(0)
                    st.image(style_file, caption="Style Reference", use_container_width=True)
                else:
                    st.image(preset_styles[selected_preset], caption="Style Reference", use_container_width=True)
            with r3:
                st.image(result_pil, caption="Stylized Output", use_container_width=True)

            buf = io.BytesIO()
            result_pil.save(buf, format="PNG")
            buf.seek(0)
            st.download_button(
                "Download Stylized Image",
                data=buf,
                file_name="stylized_output.png",
                mime="image/png",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)

# ─────────────────────────────────────────────
# THEORY SECTION
# ─────────────────────────────────────────────
st.markdown('<hr class="nst-divider">', unsafe_allow_html=True)

with st.expander("How Neural Style Transfer works"):
    st.markdown("""
<div class="theory-block">
<h4>Technical Overview</h4>
<p>
Neural Style Transfer uses a pretrained VGG19 convolutional neural network to separately extract
<em>content representations</em> and <em>style representations</em> from two images, then
synthesizes a new image that combines both.
</p>
<ul>
<li><strong>Content Loss</strong> — Compares deep feature maps at <code>conv_4</code> between the generated
image and the content image, enforcing structural and spatial fidelity.</li>
<li><strong>Style Loss</strong> — Computes differences in <strong>Gram matrices</strong> across five
convolutional layers. Gram matrices capture inter-channel feature correlations, encoding texture,
color, and brushwork without regard to spatial position.</li>
<li><strong>Gram Matrix</strong> — For a feature map of shape <code>(C, H, W)</code>, reshape to
<code>(C, H&times;W)</code> and compute <code>G = F &times; F&sup1;</code>, yielding a
<code>C&times;C</code> texture descriptor.</li>
<li><strong>L-BFGS Optimizer</strong> — Iteratively updates the generated image pixels to minimize
<code>total&nbsp;loss&nbsp;=&nbsp;style_weight&nbsp;&times;&nbsp;style_loss&nbsp;+&nbsp;content_weight&nbsp;&times;&nbsp;content_loss</code>.</li>
</ul>
<table>
<tr><th>Purpose</th><th>VGG19 Layers</th></tr>
<tr><td>Content representation</td><td><code>conv_4</code></td></tr>
<tr><td>Style representation</td><td><code>conv_1, conv_2, conv_3, conv_4, conv_5</code></td></tr>
</table>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(
    '<div class="nst-footer">'
    'Experiment 06&ensp;&middot;&ensp;LLM &amp; Generative AI Lab&ensp;&middot;&ensp;'
    'Fr. Conceicao Rodrigues College of Engineering&ensp;&middot;&ensp;'
    'Dept. of Computer Engineering'
    '</div>',
    unsafe_allow_html=True
)
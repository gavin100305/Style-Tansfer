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
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400&display=swap');

/* ── Root tokens ── */
:root {
    --bg:        #0e0e0f;
    --surface:   #161618;
    --border:    #2a2a2e;
    --muted:     #5a5a62;
    --body:      #c8c8d0;
    --bright:    #f0f0f4;
    --accent:    #c8a96e;
    --accent-dim:#7a6540;
    --danger:    #c0504a;
    --success:   #4a9e6e;
    --radius:    6px;
}

/* ── Global resets ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--body);
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Page header ── */
.nst-header {
    padding: 3rem 0 2rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
}
.nst-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 600;
    color: var(--bright);
    letter-spacing: -0.02em;
    margin: 0 0 0.4rem 0;
    line-height: 1.15;
}
.nst-header p {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    font-size: 0.95rem;
    color: var(--muted);
    margin: 0;
    letter-spacing: 0.02em;
}

/* ── Section labels ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.75rem;
    display: block;
}

/* ── Cards / panels ── */
.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    height: 100%;
}

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
}

/* ── Result label row ── */
.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    text-align: center;
    margin-top: 0.5rem;
}

/* ── Theory block ── */
.theory-block {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.75rem;
    margin-top: 1rem;
}
.theory-block h4 {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    color: var(--bright);
    margin: 0 0 1rem 0;
    font-weight: 400;
}
.theory-block p, .theory-block li {
    font-size: 0.875rem;
    color: var(--body);
    line-height: 1.7;
}
.theory-block code {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    background: #1e1e22;
    color: var(--accent);
    padding: 1px 5px;
    border-radius: 3px;
}
.theory-block table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 1rem;
    font-size: 0.82rem;
}
.theory-block th {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--border);
    text-align: left;
    font-weight: 400;
}
.theory-block td {
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--border);
    color: var(--body);
}

/* ── Footer ── */
.nst-footer {
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    color: var(--muted);
    text-align: center;
}

/* ── Streamlit widget overrides ── */
[data-testid="stFileUploader"] {
    background: #111113 !important;
    border: 1px dashed var(--border) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-dim) !important;
}

/* Buttons */
[data-testid="baseButton-primary"] {
    background: var(--accent) !important;
    color: #0e0e0f !important;
    border: none !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.06em !important;
    border-radius: var(--radius) !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.15s ease !important;
}
[data-testid="baseButton-primary"]:hover {
    opacity: 0.85 !important;
}
[data-testid="baseButton-secondary"] {
    background: transparent !important;
    color: var(--body) !important;
    border: 1px solid var(--border) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    border-radius: var(--radius) !important;
}
[data-testid="baseButton-secondary"]:hover {
    border-color: var(--accent-dim) !important;
    color: var(--accent) !important;
}

/* Selectbox, radio, slider */
[data-testid="stSelectbox"] > div > div,
[data-testid="stRadio"] label,
[data-testid="stSlider"] label,
[data-testid="stSelectSlider"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    color: var(--body) !important;
}

/* Sidebar labels */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.83rem !important;
    color: var(--body) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--bright) !important;
    font-size: 1rem !important;
    font-weight: 400 !important;
}

/* Info / error / success boxes */
[data-testid="stAlert"] {
    background: var(--surface) !important;
    border-radius: var(--radius) !important;
    border-left-width: 3px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
}

/* Progress bar */
[data-testid="stProgress"] > div > div {
    background: var(--accent) !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stExpander"] summary {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    color: var(--body) !important;
}

/* Image captions */
[data-testid="caption"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    text-align: center !important;
}

/* Status text */
.status-text {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
}

/* Sidebar device badge */
.device-badge {
    display: inline-block;
    padding: 0.25rem 0.6rem;
    border-radius: 3px;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.device-gpu  { background: #1a3a2a; color: #4a9e6e; border: 1px solid #2a5a3a; }
.device-cpu  { background: #2a2010; color: #c8a96e; border: 1px solid #4a3820; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Parameters")
    st.markdown("---")

    if device.type == "cuda":
        st.markdown('<span class="device-badge device-gpu">GPU — Accelerated</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="device-badge device-cpu">CPU — May be slow</span>', unsafe_allow_html=True)

    st.markdown("")

    image_size = st.selectbox(
        "Resolution",
        options=[128, 256, 512],
        index=0,
        help="Smaller = faster. 128px recommended for CPU."
    )

    num_steps = st.slider(
        "Optimization Steps",
        min_value=100, max_value=600, value=200, step=50
    )

    style_weight = st.select_slider(
        "Style Weight",
        options=[1e4, 1e5, 5e5, 1e6],
        value=1e5,
        format_func=lambda x: f"{x:.0e}"
    )

    content_weight = st.slider(
        "Content Weight",
        min_value=1, max_value=10, value=1
    )

    st.markdown("---")
    st.markdown(
        '<p style="font-family:\'DM Mono\',monospace;font-size:0.68rem;color:#5a5a62;line-height:1.6;">'
        'Higher style weight = more artistic distortion.<br>'
        'Higher content weight = more structural fidelity.'
        '</p>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="nst-header">
    <h1>Neural Style Transfer</h1>
    <p>Reimagine any photograph through the lens of artistic masterworks &mdash; powered by VGG19</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPER FUNCTIONS (unchanged)
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
    import requests
    import tempfile

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
            with torch.no_grad():
                input_img.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            loss = style_weight   * sum(sl.loss for sl in style_losses) + \
                   content_weight * sum(cl.loss for cl in content_losses)
            loss.backward()
            run[0] += 1
            if run[0] % 10 == 0:
                progress_bar.progress(min(int(run[0] / num_steps * 100), 100))
                status_text.markdown(
                    f'<p class="status-text">Step {run[0]} / {num_steps} &nbsp;&mdash;&nbsp; Loss: {loss.item():.4f}</p>',
                    unsafe_allow_html=True
                )
            return loss
        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)
    return input_img

# ─────────────────────────────────────────────
# IMAGE INPUT SECTION
# ─────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<span class="section-label">01 — Content Image</span>', unsafe_allow_html=True)
    content_file = st.file_uploader(
        "Upload the photograph you want to stylize",
        type=["jpg", "jpeg", "png"],
        key="content",
        label_visibility="collapsed"
    )
    st.markdown(
        '<p style="font-size:0.78rem;color:#5a5a62;margin-top:0.4rem;">'
        'JPG or PNG — this is the image whose structure will be preserved</p>',
        unsafe_allow_html=True
    )
    if content_file:
        st.image(content_file, caption="Content", width=420)

with col2:
    st.markdown('<span class="section-label">02 — Style Reference</span>', unsafe_allow_html=True)

    preset_styles = {
        "Van Gogh — Starry Night": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
        "Picasso — Weeping Woman": "https://upload.wikimedia.org/wikipedia/en/1/14/Picasso_The_Weeping_Woman_Tate_identifier_T05010_10.jpg",
        "Monet — Water Lilies":    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/Claude_Monet_-_Water_Lilies_-_1906%2C_Ryerson.jpg/1280px-Claude_Monet_-_Water_Lilies_-_1906%2C_Ryerson.jpg",
    }

    style_option = st.radio(
        "Style source",
        ["Upload your own", "Use preset artwork"],
        horizontal=True,
        label_visibility="collapsed"
    )

    style_file = None
    selected_preset = None

    if style_option == "Upload your own":
        style_file = st.file_uploader(
            "Upload artwork image",
            type=["jpg", "jpeg", "png"],
            key="style",
            label_visibility="collapsed"
        )
        st.markdown(
            '<p style="font-size:0.78rem;color:#5a5a62;margin-top:0.4rem;">'
            'Any artwork or texture image — its visual character will be transferred</p>',
            unsafe_allow_html=True
        )
        if style_file:
            st.image(style_file, caption="Style Reference", width=420)
    else:
        selected_preset = st.selectbox(
            "Select artwork",
            list(preset_styles.keys()),
            label_visibility="collapsed"
        )
        st.image(preset_styles[selected_preset], caption=selected_preset, width=420)

# ─────────────────────────────────────────────
# GENERATE BUTTON
# ─────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)

generate = st.button(
    "Generate Stylized Image",
    type="primary",
    use_container_width=True
)

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
                f"at {image_size}px resolution. This may take several minutes on CPU."
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

            # ── Result display ──
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
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

            # ── Download ──
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
st.markdown('<hr class="divider">', unsafe_allow_html=True)

with st.expander("How Neural Style Transfer works"):
    st.markdown("""
<div class="theory-block">
<h4>Technical Overview</h4>
<p>
Neural Style Transfer (NST) uses a pretrained VGG19 convolutional neural network to separately
extract <em>content representations</em> and <em>style representations</em> from two images,
then synthesizes a new image that blends both.
</p>
<ul>
<li><strong>Content Loss</strong> — Compares deep feature maps at <code>conv_4</code> between
the generated image and the content image. This enforces structural and spatial fidelity.</li>
<li><strong>Style Loss</strong> — Computes the difference in <strong>Gram matrices</strong>
across five convolutional layers. Gram matrices capture inter-channel feature correlations —
encoding texture, color, and brushwork without regard to spatial arrangement.</li>
<li><strong>Gram Matrix</strong> — For a feature map of shape <code>(C, H, W)</code>, reshape
to <code>(C, H&times;W)</code>, then compute <code>G = F &times; F&sup1;</code>. This produces
a <code>C&times;C</code> texture descriptor.</li>
<li><strong>L-BFGS Optimizer</strong> — Iteratively updates the generated image to minimize
<code>total loss = style_weight &times; style_loss + content_weight &times; content_loss</code>.</li>
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
    'Experiment 06 &nbsp;&middot;&nbsp; LLM &amp; Generative AI Lab &nbsp;&middot;&nbsp; '
    'Fr. Conceicao Rodrigues College of Engineering &nbsp;&middot;&nbsp; '
    'Dept. of Computer Engineering'
    '</div>',
    unsafe_allow_html=True
)
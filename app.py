"""
╔══════════════════════════════════════════════════════════════════╗
║          EXO HUNTER AI  —  Streamlit Web App           ║
║          1D-CNN on Kepler Light Curves             ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings, os, time, math
import base64
import mimetypes

warnings.filterwarnings("ignore")


def display_embedded_image(image_path, caption=None):
    """Display an image as a Base64 data URI, bypassing Streamlit's media endpoint."""
    if not os.path.isfile(image_path):
        st.warning(f"Image not found: {image_path}")
        return

    mime_type = mimetypes.guess_type(image_path)[0] or "image/png"

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("ascii")

    st.markdown(
        f"""
        <img
            src="data:{mime_type};base64,{encoded_image}"
            style="width:100%; height:auto;"
            alt="Scientific visualization"
        />
        """,
        unsafe_allow_html=True,
    )

    if caption:
        st.caption(caption)


# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Exo Hunter AI | Exoplanet Transit Detection",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────────
INPUT_LEN        = 3197
THRESHOLD        = 0.6914
EB_RATIO_THRESH  = 0.50
MODEL_PATH       = "models/cnn_model_week6.keras"

DEMO_STARS = {
    "KIC 10666592 — Kepler-2b  (expect PLANET ≈0.977)": "10666592",
    "KIC 11446443 — Kepler-1b  (expect PLANET ≈0.977)": "11446443",
    "KIC 9941662  — Kepler-13b (expect MISSED, score<0.6914)": "9941662",
    "KIC 3335816  — Equal-eclipse EB (expect FLAGGED)":  "3335816",
}

# ─────────────────────────────────────────────────────────────────
# GLOBAL CSS / THEME
# ─────────────────────────────────────────────────────────────────
STAR_CSS = """
<style>
/* ── fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;900&family=Share+Tech+Mono&family=DM+Sans:wght@300;400;500&display=swap');

/* ── base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #020408 !important;
    color: #c8d8f0 !important;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background: #050b14 !important;
    border-right: 1px solid #0d2240;
}
[data-testid="stSidebar"] * { color: #8aafd4 !important; }
h1, h2, h3, h4 { font-family: 'Orbitron', monospace !important; }

/* ── star canvas ── */
#star-canvas {
    position: fixed; top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: -1; pointer-events: none;
}

/* ── metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #06111f 0%, #0a1c30 100%);
    border: 1px solid #0d3055;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: border-color .3s;
}
.metric-card:hover { border-color: #2575fc; }
.metric-val {
    font-family: 'Orbitron', monospace;
    font-size: 2rem; font-weight: 900;
    background: linear-gradient(90deg, #2575fc, #6a11cb);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-label { font-size: .75rem; letter-spacing: .1em; color: #5a7fa8; margin-top:.3rem; }

/* ── result boxes ── */
.result-planet {
    background: linear-gradient(135deg,#001a40,#002d6e);
    border: 2px solid #2575fc;
    border-radius: 16px; padding: 2rem; text-align: center;
    box-shadow: 0 0 40px #2575fc44;
}
.result-none {
    background: linear-gradient(135deg,#0f0a00,#241800);
    border: 2px solid #f5a623;
    border-radius: 16px; padding: 2rem; text-align: center;
    box-shadow: 0 0 30px #f5a62322;
}
.result-eb {
    background: linear-gradient(135deg,#1a0010,#330020);
    border: 2px solid #e74c3c;
    border-radius: 16px; padding: 2rem; text-align: center;
    box-shadow: 0 0 30px #e74c3c33;
}
.big-emoji { font-size: 4rem; line-height: 1.1; }
.result-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.6rem; font-weight: 900; margin:.5rem 0;
}
.conf-score {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.5rem; letter-spacing: .05em;
}
.tier-badge {
    display:inline-block; padding: .25rem .9rem;
    border-radius: 999px; font-size: .8rem;
    font-family:'Orbitron',monospace; letter-spacing:.08em;
    margin-top:.5rem;
}
.tier-high   { background:#163366; border:1px solid #2575fc; color:#7ab3ff; }
.tier-med    { background:#1a1200; border:1px solid #f5a623; color:#f5c87a; }
.tier-low    { background:#160606; border:1px solid #8b0000; color:#e87070; }

/* ── step card ── */
.step-card {
    background: #060f1c;
    border-left: 3px solid #2575fc;
    border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: .8rem;
}
.step-num {
    font-family:'Orbitron',monospace; font-size:1.4rem;
    color:#2575fc; font-weight:900;
}

/* ── timeline ── */
.timeline-item {
    border-left: 2px solid #0d2240;
    padding: .6rem 1rem .6rem 1.4rem;
    margin-bottom:.5rem; position:relative;
}
.timeline-item::before {
    content:''; position:absolute; left:-5px; top:50%;
    transform:translateY(-50%);
    width:8px; height:8px; border-radius:50%;
    background:#2575fc;
}
.week-badge {
    font-family:'Orbitron',monospace; font-size:.65rem;
    color:#2575fc; letter-spacing:.1em;
}

/* ── warning / info banners ── */
.warn-box {
    background:#1a0f00; border:1px solid #f5a623;
    border-radius:8px; padding:.8rem 1rem;
    color:#c8a860; font-size:.85rem; margin:.5rem 0;
}
.info-box {
    background:#00101a; border:1px solid #0d4070;
    border-radius:8px; padding:.8rem 1rem;
    color:#6aadcc; font-size:.85rem; margin:.5rem 0;
}

/* ── code-mono spans ── */
.mono { font-family:'Share Tech Mono',monospace; color:#5af; }

/* ── sidebar nav active ── */
div[data-testid="stRadio"] label { cursor:pointer; }

/* ── plotly chart bg ── */
.js-plotly-plot .plotly { border-radius:12px; }

/* ── pulse animation ── */
@keyframes pulse-glow {
    0%,100% { box-shadow: 0 0 20px #2575fc44; }
    50%      { box-shadow: 0 0 50px #2575fc88; }
}
.planet-pulse { animation: pulse-glow 2.5s ease-in-out infinite; }

/* ── scrollbar ── */
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:#020408; }
::-webkit-scrollbar-thumb { background:#0d2240; border-radius:3px; }
</style>

<!-- Animated star field -->
<canvas id="star-canvas"></canvas>
<script>
(function(){
    var c = document.getElementById('star-canvas');
    if(!c) return;
    var ctx = c.getContext('2d');
    var stars = [];
    function resize(){ c.width=window.innerWidth; c.height=window.innerHeight; }
    resize();
    window.addEventListener('resize', resize);
    for(var i=0;i<220;i++){
        stars.push({
            x: Math.random()*window.innerWidth,
            y: Math.random()*window.innerHeight,
            r: Math.random()*1.4+.2,
            a: Math.random(),
            da: (Math.random()-.5)*.008
        });
    }
    function draw(){
        ctx.clearRect(0,0,c.width,c.height);
        stars.forEach(function(s){
            s.a += s.da;
            if(s.a<=0||s.a>=1) s.da*=-1;
            ctx.beginPath();
            ctx.arc(s.x,s.y,s.r,0,Math.PI*2);
            ctx.fillStyle='rgba(160,200,255,'+s.a+')';
            ctx.fill();
        });
        requestAnimationFrame(draw);
    }
    draw();
})();
</script>
"""

st.markdown(STAR_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# CACHED RESOURCE: MODEL
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CNN model …")
def load_model():
    """Load Keras model once and cache across sessions."""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, None
    except FileNotFoundError:
        return None, f"Model file not found at `{MODEL_PATH}`."
    except Exception as e:
        return None, str(e)

# ─────────────────────────────────────────────────────────────────
# CACHED DATA: LIGHT CURVE DOWNLOAD
# ─────────────────────────────────────────────────────────────────
class _LC:
    """
    Lightweight pickle-safe container that mimics the lightkurve LightCurve
    interface used by preprocess() and run_eb_check().
    Stores only plain numpy arrays — fully serialisable by st.cache_data.
    """
    def __init__(self, time_arr: np.ndarray, flux_arr: np.ndarray):
        self._time = time_arr
        self._flux = flux_arr

    @property
    def flux(self):
        return self._flux

    class _TimeWrapper:
        """Mimics lc.time.value access pattern."""
        def __init__(self, arr):
            self.value = arr

    @property
    def time(self):
        return self._TimeWrapper(self._time)


def _stitch_chunks(time_chunks, flux_chunks):
    """
    Normalise each quarter to its median (replicates lightkurve stitch),
    concatenate, sort by time, remove duplicate cadences at boundaries.
    Returns (all_time, all_flux) as float64 numpy arrays.
    """
    norm_chunks = []
    for f_chunk in flux_chunks:
        valid = f_chunk[~np.isnan(f_chunk)]
        q_med = np.median(valid) if len(valid) > 0 else 1.0
        if q_med == 0 or np.isnan(q_med):
            q_med = 1.0
        norm_chunks.append(f_chunk / q_med)

    all_time = np.concatenate(time_chunks)
    all_flux = np.concatenate(norm_chunks)
    idx      = np.argsort(all_time)
    all_time = all_time[idx]
    all_flux = all_flux[idx]
    _, uid   = np.unique(all_time, return_index=True)
    return all_time[uid], all_flux[uid]


@st.cache_data(show_spinner=False, ttl=86400)
def download_light_curve(kic_id: str):
    """
    Fast Kepler download strategy:
      1. ONE search_lightcurve call to get the full result table (quarters 1-8)
      2. ONE download_all() on just those results — lightkurve batches the
         MAST requests internally, much faster than 8 individual searches.
      3. Median-normalise + stitch manually (pickle-safe numpy arrays).

    Cached 24h — second run is instant.
    Returns (_LC, error_or_None, n_quarters, n_cadences)
    """
    import lightkurve as lk

    # ── Single MAST search (one round trip) ──
    try:
        search = lk.search_lightcurve(
            f"KIC {kic_id}",
            mission="Kepler",
            author="Kepler",
            cadence="long",
            quarter=list(range(1, 9)),   # filter server-side — faster
        )
    except Exception as e:
        msg = str(e)
        if "timeout" in msg.lower() or "timed out" in msg.lower():
            return None, "MAST search timed out — please retry.", 0, 0
        return None, f"MAST error: {msg}", 0, 0

    if len(search) == 0:
        return None, f"KIC {kic_id} not found in the Kepler archive. Check the ID.", 0, 0

    # ── Single batched download (lightkurve handles the FITS fetching) ──
    try:
        lc_col = search.download_all(flux_column="pdcsap_flux", quality_bitmask="default")
    except Exception as e:
        msg = str(e)
        if "timeout" in msg.lower() or "timed out" in msg.lower():
            return None, "MAST download timed out — please retry.", 0, 0
        return None, f"Download error: {msg}", 0, 0

    if lc_col is None or len(lc_col) == 0:
        return None, "No PDCSAP flux available for this star.", 0, 0

    # ── Extract arrays per quarter then stitch (no lightkurve object in cache) ──
    time_chunks, flux_chunks = [], []
    for lc in lc_col:
        try:
            t = np.array(lc.time.value, dtype=np.float64)
            f = np.array(lc.flux,       dtype=np.float64)
            if len(t) > 10:
                time_chunks.append(t)
                flux_chunks.append(f)
        except Exception:
            continue

    if not time_chunks:
        return None, "Could not extract flux arrays from downloaded data.", 0, 0

    all_time, all_flux = _stitch_chunks(time_chunks, flux_chunks)
    return _LC(all_time, all_flux), None, len(time_chunks), len(all_time)

# ─────────────────────────────────────────────────────────────────
# PREPROCESSING PIPELINE  (exact — do not alter)
# ─────────────────────────────────────────────────────────────────
def preprocess(lc):
    """
    Week 2 pipeline:
    1. NaN interpolation
    2. Median subtraction
    3. Best-variance segment (sliding window, step=INPUT_LEN//4)
    4. L2 normalisation
    5. Gaussian smoothing σ=10
    6. float32 cast
    Returns (segment, flux_raw_full, time_full, seg_start_idx) or raises ValueError.
    """
    from scipy.ndimage import gaussian_filter1d

    flux_raw = np.array(lc.flux, dtype=float)
    time_raw = np.array(lc.time.value, dtype=float)

    # 1. NaN interpolation
    nans = np.isnan(flux_raw)
    if nans.all():
        raise ValueError("All flux values are NaN — insufficient signal.")
    x_idx = np.arange(len(flux_raw))
    flux = np.interp(x_idx, x_idx[~nans], flux_raw[~nans])

    # 2. Median subtraction
    flux = flux - np.median(flux)

    if len(flux) < INPUT_LEN:
        raise ValueError(
            f"Light curve too short ({len(flux)} pts). "
            f"Need at least {INPUT_LEN} cadences."
        )

    # 3. Best-variance segment
    step = INPUT_LEN // 4
    best_var, best_start = -1.0, 0
    for s in range(0, len(flux) - INPUT_LEN + 1, step):
        seg_var = np.var(flux[s : s + INPUT_LEN])
        if seg_var > best_var:
            best_var, best_start = seg_var, s
    segment = flux[best_start : best_start + INPUT_LEN]

    if np.all(segment == 0):
        raise ValueError("Preprocessed segment is all zeros — insufficient signal.")

    # 4. L2 normalisation
    norm = np.linalg.norm(segment)
    if norm == 0:
        raise ValueError("L2 norm is zero — insufficient signal.")
    segment = segment / norm

    # 5. Gaussian smoothing σ=10
    segment = gaussian_filter1d(segment, sigma=10)

    # 6. float32
    segment = segment.astype(np.float32)

    return segment, flux, time_raw, best_start

# ─────────────────────────────────────────────────────────────────
# EB REJECTION PIPELINE
# ─────────────────────────────────────────────────────────────────
def run_eb_check(lc, period: float):
    """
    Full EB rejection pipeline.
    Returns dict: {is_eb, primary_depth, secondary_depth, ratio, method, skipped, reason}

    Bug fixes vs original:
    1. Primary depth gate: must exceed 5x OOT scatter, not just > 0
       (prevents noise spikes from being treated as real eclipses)
    2. Secondary noise gate raised to 3x OOT scatter (was 2x)
       (tighter gate — real EB secondaries are well above noise)
    3. Absolute minimum depth guard: primary must be > 1e-4 in normalised units
       (catches near-flat LCs where fold finds noise as "primary")
    4. Ratio guard: only flag EB if BOTH depths are physically meaningful
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.stats import sigmaclip

    result = dict(is_eb=False, primary_depth=None, secondary_depth=None,
                  ratio=None, method=None, skipped=False, reason="")

    try:
        flux = np.array(lc.flux, dtype=float)
        time = np.array(lc.time.value, dtype=float)

        nans = np.isnan(flux)
        flux = np.interp(np.arange(len(flux)), np.arange(len(flux))[~nans], flux[~nans])
        flux -= np.median(flux)

        # Normalise by flux scale so depth thresholds are scale-independent
        flux_scale = np.std(flux)
        if flux_scale == 0:
            result["skipped"] = True
            result["reason"]  = "Flux is flat — cannot perform EB check."
            return result

        def fold_and_check(p):
            phase = ((time - time[0]) % p) / p

            # ── coarse 100-bin fold to find t0 robustly ──
            bins_c   = np.linspace(0, 1, 101)
            idx_c    = np.digitize(phase, bins_c) - 1
            binned_c = np.array([
                np.median(flux[idx_c == i]) if np.any(idx_c == i) else 0.0
                for i in range(100)
            ])
            t0_bin   = np.argmin(binned_c)
            t0_phase = bins_c[t0_bin] + 0.005

            # ── centre primary at phase 0.5 ──
            phase_s = (phase - t0_phase + 0.5) % 1.0

            # ── OOT mask: exclude primary (0.4–0.6) and secondary (0.9–1.0 / 0.0–0.1) ──
            oot_mask = (
                ((phase_s < 0.4) | (phase_s > 0.6)) &
                ((phase_s > 0.1) & (phase_s < 0.9))
            )
            if oot_mask.sum() < 20:
                return None

            oot_vals, _, _ = sigmaclip(flux[oot_mask], low=3, high=3)
            if len(oot_vals) == 0:
                return None
            oot_med = np.median(oot_vals)
            flux_n  = flux - oot_med

            # ── 300-bin phase fold + light smoothing ──
            bins300 = np.linspace(0, 1, 301)
            idx300  = np.digitize(phase_s, bins300) - 1
            binned  = np.array([
                np.median(flux_n[idx300 == i]) if np.any(idx300 == i) else 0.0
                for i in range(300)
            ])
            binned = gaussian_filter1d(binned, sigma=1)

            # ── OOT noise floor (bins away from both eclipse windows) ──
            oot_bin_idx = list(range(15, 90)) + list(range(110, 190)) + list(range(210, 285))
            oot_scatter = np.std(binned[oot_bin_idx])
            if oot_scatter == 0:
                return None

            # ── Primary depth: deepest point in phase 0.4–0.6 (bin 120–180) ──
            primary_depth = -binned[120:180].min()

            # ── Gate 1: primary must be physically real (>5σ above OOT noise) ──
            if primary_depth < 5 * oot_scatter:
                return None  # no credible primary eclipse

            # ── Gate 2: absolute minimum depth (>1e-4 in flux units) ──
            if primary_depth < 1e-4:
                return None  # depth too shallow to be meaningful

            # ── Secondary depth: deepest point near phase 0.0/1.0 (bins 0–30 + 270–300) ──
            sec_bins      = np.concatenate([binned[:30], binned[270:]])
            secondary_depth = -sec_bins.min()

            # ── Gate 3: secondary must exceed 3x OOT scatter to be real ──
            if secondary_depth < 3 * oot_scatter:
                secondary_depth = 0.0

            ratio = secondary_depth / primary_depth
            return primary_depth, secondary_depth, ratio

        # ── Try 1P fold first ──
        res = fold_and_check(period)
        if res is None:
            # ── Fallback: 2P fold catches equal-eclipse EBs ──
            res = fold_and_check(period * 2)
            if res is None:
                result["skipped"] = True
                result["reason"]  = "No credible primary eclipse detected (signal below 5σ noise floor)."
                return result
            result["method"] = "2P fold (equal-eclipse EB check)"
        else:
            result["method"] = "Standard 1P fold"

        primary_depth, secondary_depth, ratio = res
        result["primary_depth"]   = primary_depth
        result["secondary_depth"] = secondary_depth
        result["ratio"]           = ratio
        # Only flag EB if secondary is also physically present AND ratio exceeds threshold
        result["is_eb"] = (secondary_depth > 0) and (ratio > EB_RATIO_THRESH)

    except Exception as e:
        result["skipped"] = True
        result["reason"]  = f"EB check error: {e}"

    return result

# ─────────────────────────────────────────────────────────────────
# CONFIDENCE TIER
# ─────────────────────────────────────────────────────────────────
def get_confidence_tier(score: float):
    if score > 0.50:
        return "HIGH", "tier-high", "⬛ HIGH CONFIDENCE"
    elif score >= 0.05:
        return "MEDIUM", "tier-med", "⬛ MEDIUM — FOLLOW UP"
    else:
        return "LOW", "tier-low", "⬛ LOW CONFIDENCE"

# ─────────────────────────────────────────────────────────────────
# PLOTLY THEME HELPER
# ─────────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(2,4,8,0)",
    plot_bgcolor="rgba(6,15,28,0.9)",
    font=dict(family="DM Sans", color="#8aafd4"),
    xaxis=dict(gridcolor="#0d2240", zerolinecolor="#0d2240"),
    yaxis=dict(gridcolor="#0d2240", zerolinecolor="#0d2240"),
    margin=dict(l=50, r=30, t=50, b=50),
)

# ─────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1.2rem 0 .5rem;'>
      <div style='font-family:Orbitron,monospace; font-size:1.1rem;
                  font-weight:900; color:#2575fc; letter-spacing:.12em;'>
        🪐 EXO HUNTER AI
      </div>
      <div style='font-size:.65rem; letter-spacing:.15em; color:#2d4d6e; margin-top:.3rem;'>
        KEPLER · 1D-CNN · TRANSIT DETECTION
      </div>
    </div>
    <hr style='border-color:#0d2240; margin:.5rem 0 1rem;'>
    """, unsafe_allow_html=True)

    PAGE = st.radio(
        "Navigate",
        ["🏠  Home", "🔭  Detect", "📊  Performance", "📖  About"],
        label_visibility="collapsed",
    )
    st.markdown("<hr style='border-color:#0d2240; margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.7rem; color:#2d4d6e; line-height:1.8;'>
      Model: <span style='color:#4a80b0'>1D CNN</span><br>
      Threshold: <span style='color:#4a80b0'>0.6914</span><br>
      AUC: <span style='color:#4a80b0'>0.9628</span><br>
      EB Filter: <span style='color:#4a80b0'>Active</span>
    </div>
    """, unsafe_allow_html=True)

page = PAGE.split("  ", 1)[-1].strip()

# ═════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═════════════════════════════════════════════════════════════════
if page == "Home":
    # ── Hero ──
    st.markdown("""
    <div style='text-align:center; padding: 3rem 0 2rem;'>
      <div style='font-family:Orbitron,monospace; font-size:.8rem; letter-spacing:.3em;
                  color:#2575fc; margin-bottom:1rem;'>
        KEPLER SPACE TELESCOPE · 1D-CNN · TRANSIT PHOTOMETRY
      </div>
      <h1 style='font-size:clamp(2rem,5vw,3.5rem); font-weight:900; margin:0;
                 background:linear-gradient(135deg,#ffffff,#7ab3ff,#6a11cb);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        EXO HUNTER AI
      </h1>
      <div style='font-family:"DM Sans",sans-serif; font-size:1.1rem; color:#5a7fa8;
                  margin-top:1rem; max-width:640px; margin-left:auto; margin-right:auto;'>
        A 1D Convolutional Neural Network trained on 150,000+ Kepler stellar light curves,
        hunting for the faint dimming signatures of worlds beyond our solar system.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Key Stats ──
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("0.9628", "Benchmark ROC-AUC"),
        ("0.6933", "Wild-data ROC-AUC"),
        ("0.6914", "Decision Threshold"),
        ("0%",     "Broader-catalog detection*"),
    ]
    for col, (val, label) in zip([c1,c2,c3,c4], stats):
        col.markdown(f"""
        <div class='metric-card'>
          <div class='metric-val'>{val}</div>
          <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.caption("*The saved broader-catalog evaluation reported 0% detection on its tested sample; this is an important limitation, not a universal estimate for all catalogs.")
    st.markdown("<div style='height:2.5rem'></div>", unsafe_allow_html=True)

    # ── How It Works ──
    st.markdown("""
    <h2 style='text-align:center; font-size:1.2rem; letter-spacing:.15em; color:#4a80b0;'>
      HOW IT WORKS
    </h2>""", unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    steps = [
        ("01", "📥 Acquire", "Enter a Kepler Input Catalogue (KIC) number. The app auto-downloads the PDCSAP flux light curve — up to 8 quarters — from the MAST archive via <em>lightkurve</em>."),
        ("02", "⚙️ Preprocess", "The preprocessing pipeline handles missing values, selects a 3,197-cadence segment, normalises flux, and applies Gaussian smoothing (σ=10)."),
        ("03", "🧠 Predict", "The trained 1D-CNN scores the segment [0,1]. Scores above <strong>0.6914</strong> trigger planet-candidate status. An EB rejection filter checks for secondary eclipses."),
    ]
    for col, (num, title, body) in zip([s1,s2,s3], steps):
        col.markdown(f"""
        <div class='step-card'>
          <div class='step-num'>{num}</div>
          <div style='font-family:Orbitron,monospace; font-size:.85rem; color:#7ab3ff;
                      margin:.3rem 0;'>{title}</div>
          <div style='font-size:.85rem; color:#6a8db0; line-height:1.6;'>{body}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

    # ── What the model detects ──
    col_l, col_r = st.columns([1.1, 1])

    with col_l:
        st.markdown("""
        <h3 style='font-family:Orbitron,monospace; font-size:.9rem; letter-spacing:.12em;
                   color:#4a80b0;'>DETECTION CAPABILITIES</h3>""", unsafe_allow_html=True)

        capabilities = [
            ("✅", "Hot Jupiters (SNR ≥ 200)", "93% detection rate"),
            ("✅", "Medium-SNR planets (50–200)", "Partial detection"),
            ("⚠️", "Low-SNR planets (< 50)",     "7% detection rate"),
            ("⚠️", "Stellar variability",          "Scores near zero"),
            ("✅", "Eclipsing binaries",            "100% catch rate (6/6)"),
            ("✅", "Equal-eclipse EBs",             "2P fold fallback"),
        ]
        for icon, feat, note in capabilities:
            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:.8rem;
                        padding:.5rem 0; border-bottom:1px solid #0a1929;'>
              <span style='font-size:1.1rem'>{icon}</span>
              <div>
                <div style='font-size:.85rem; color:#8aafd4'>{feat}</div>
                <div style='font-size:.72rem; color:#3d5f80'>{note}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <h3 style='font-family:Orbitron,monospace; font-size:.9rem; letter-spacing:.12em;
                   color:#4a80b0;'>MODEL ARCHITECTURE</h3>""", unsafe_allow_html=True)

        arch_items = [
            ("Input",  "Shape (1, 3197, 1) — PDCSAP flux segment"),
            ("Conv1",  "64 filters, kernel 3, ReLU + MaxPool"),
            ("Conv2",  "128 filters, kernel 3, ReLU + MaxPool"),
            ("Conv3",  "256 filters, kernel 3, ReLU + GlobalAvgPool"),
            ("Dense",  "128 units, ReLU + Dropout 0.5"),
            ("Output", "1 unit, Sigmoid → planet probability"),
        ]
        for layer, desc in arch_items:
            st.markdown(f"""
            <div style='display:flex; gap:1rem; padding:.45rem 0;
                        border-bottom:1px solid #0a1929; align-items:baseline;'>
              <span style='font-family:"Share Tech Mono",monospace; color:#2575fc;
                           font-size:.78rem; min-width:52px;'>{layer}</span>
              <span style='font-size:.78rem; color:#6a8db0;'>{desc}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
      🚀 <strong>Try it:</strong> Navigate to the <em>Detect</em> page and enter a KIC number —
      or pick one of the demo stars from the dropdown. Results include a confidence score,
      light-curve plot with transit dip markers, and an eclipsing-binary check.
    </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════
# PAGE 2 — DETECT
# ═════════════════════════════════════════════════════════════════
elif page == "Detect":
    st.markdown("""
    <h1 style='font-family:Orbitron,monospace; font-size:1.6rem; font-weight:900;
               color:#7ab3ff; letter-spacing:.1em; margin-bottom:.3rem;'>
      🔭 DETECT EXOPLANETS
    </h1>
    <div style='color:#4a6a8a; font-size:.85rem; margin-bottom:1.5rem;'>
      Enter a Kepler Input Catalogue (KIC) number to run the full detection pipeline.
    </div>""", unsafe_allow_html=True)

    # ── Load model ──
    model, model_err = load_model()

    if model_err:
        st.error(f"⚠️ Model load failed: {model_err}")
        st.info("Check that the trained CNN model file is present inside the `models/` folder at the app root.")
        st.stop()

    # ── Input area ──
    col_input, col_demo = st.columns([1.3, 1])

    with col_input:
        kic_raw = st.text_input(
            "Kepler Star ID (KIC number)",
            placeholder="e.g. 11446443",
            help="Integer ID from the Kepler Input Catalogue. Do not include 'KIC'.",
        )

    with col_demo:
        demo_choice = st.selectbox(
            "Or pick a demo star",
            ["— select —"] + list(DEMO_STARS.keys()),
            help="Example targets for quickly testing the detection pipeline.",
        )

    if demo_choice != "— select —":
        kic_raw = DEMO_STARS[demo_choice]

    # ── Known periods for ALL test stars ──
    # For EBs: use the HALF-period so that 2P fold catches equal-eclipse systems.
    # KIC 3335816 is an equal-eclipse EB with true period ~0.9077d — enter half
    # (0.4539d) so the 2P fold at 0.9077d detects both eclipses symmetrically.
    KNOWN_PERIODS = {
        "10666592": 2.4218757,   # Kepler-2b  — confirmed planet
        "11446443": 2.4706132,   # Kepler-1b  — confirmed planet
        "9941662":  0.0,          # Kepler-13b — grazing transit, score-only result (no EB check needed)
        "3335816":  0.45385,     # Equal-eclipse EB: half-period → 2P fold = 0.9077d
        "8191672":  3.0240949,   # Kepler-43b — confirmed planet
        "5357901":  3.2134120,   # Kepler-4b  — confirmed planet
        "9410930":  1.8556135,   # Kepler-41b — confirmed planet
        "10619192": 1.4857108,   # Kepler-17b — confirmed planet (active star)
        "5780885":  4.8854892,   # Kepler-7b  — confirmed planet (long period)
        "6370665":  3.3000000,   # Classic EB
        "9025971":  0.6600000,   # Contact binary EB
        "3632418":  0.0,         # Variable star — no period needed
        "6278762":  0.0,         # Delta Scuti — no period needed
    }

    # ── Auto-fill: works for both typed KIC AND demo dropdown ──
    # Normalise whatever is in the text box right now
    _typed_kic = kic_raw.strip().replace("KIC","").replace(" ","") if kic_raw else ""
    # Demo dropdown overrides typed KIC
    _active_kic = DEMO_STARS[demo_choice] if demo_choice != "— select —" else _typed_kic
    _auto_period = KNOWN_PERIODS.get(_active_kic, 0.0)
    _period_known = _active_kic in KNOWN_PERIODS

    # ── Reactive period: update session state whenever KIC changes ──
    # st.number_input ignores value= after first render, so we drive it
    # via session_state key. Whenever the active KIC changes, we update
    # the key directly → widget re-renders with the correct period.
    _period_key = "period_input_val"
    _prev_key   = "period_input_kic"

    if st.session_state.get(_prev_key) != _active_kic:
        # KIC changed — push the correct period into the widget state
        st.session_state[_period_key] = float(_auto_period)
        st.session_state[_prev_key]   = _active_kic

    col_period, col_eb_info = st.columns([1, 1])
    with col_period:
        period_input = st.number_input(
            "Orbital period (days) for EB check",
            min_value=0.0,
            step=0.0001,
            format="%.4f",
            key=_period_key,
            help="Auto-filled for known KICs. Edit manually for others. Leave 0 to skip EB check.",
        )
    with col_eb_info:
        if _period_known and _auto_period > 0:
            if _active_kic == "3335816":
                st.markdown(f"""
                <div class='info-box' style='margin-top:1.6rem; font-size:.78rem;'>
                  ⚡ <strong>Auto-filled: {_auto_period:.4f} d</strong>
                  (half-period = {_auto_period*2:.4f} d true period)<br>
                  Equal-eclipse EB — 2P fold will detect both identical eclipses.
                  CNN score may be high; EB filter should still flag it.
                </div>""", unsafe_allow_html=True)
            elif _active_kic == "6370665":
                st.markdown(f"""
                <div class='info-box' style='margin-top:1.6rem; font-size:.78rem;'>
                  ⚡ <strong>Auto-filled: {_auto_period:.4f} d</strong>
                  — classic EB. Expect 🚨 EB FLAGGED via standard 1P fold.
                </div>""", unsafe_allow_html=True)
            elif _active_kic == "9025971":
                st.markdown(f"""
                <div class='info-box' style='margin-top:1.6rem; font-size:.78rem;'>
                  ⚡ <strong>Auto-filled: {_auto_period:.4f} d</strong>
                  — short-period contact binary. Expect 🚨 EB FLAGGED.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='info-box' style='margin-top:1.6rem; font-size:.78rem;'>
                  ⚡ <strong>Auto-filled: {_auto_period:.7f} d</strong>
                  — confirmed catalog period. Edit if needed.
                </div>""", unsafe_allow_html=True)
        elif _period_known and _auto_period == 0:
            st.markdown("""
            <div class='info-box' style='margin-top:1.6rem; font-size:.78rem;'>
              ℹ️ No period for this star — EB check skipped.
              CNN score alone determines the result.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='info-box' style='margin-top:1.6rem; font-size:.78rem;'>
              ℹ️ Enter the known orbital period to enable the EB check,
              or leave at 0 to skip. Period auto-fills for known KICs.
            </div>""", unsafe_allow_html=True)

    run_btn = st.button("🚀 Run Detection Pipeline", type="primary", use_container_width=True)

    if run_btn:
        # ── validate KIC ──
        kic_raw = kic_raw.strip().replace("KIC", "").replace(" ", "")
        if not kic_raw.isdigit():
            st.error("Invalid KIC ID — must be a plain integer (e.g. 11446443).")
            st.stop()
        kic_id = kic_raw

        # ── Download — session cache → st.cache_data → MAST ──
        with st.status("📡 Contacting MAST archive …", expanded=True) as status:

            cached = st.session_state.get(f"lc_{kic_id}")
            if cached is not None:
                # Layer 1: same-session instant hit
                lc, n_quarters, n_cadences = cached
                st.write(f"⚡ Session cache hit — {n_cadences:,} cadences "
                         f"({n_quarters} quarters) — skipping download")
            else:
                st.write(f"🔍 Searching MAST for KIC {kic_id} …")
                pb = st.progress(0, text="Contacting MAST …")
                t0 = time.time()

                # Layer 2: st.cache_data (24h) → or fresh MAST download
                # download_light_curve does ONE search + ONE download_all internally
                pb.progress(20, text="Fetching light curves from MAST …")
                result = download_light_curve(kic_id)
                pb.progress(90, text="Stitching quarters …")

                lc, lc_err, n_quarters, n_cadences = result

                pb.progress(100, text="Done")
                pb.empty()
                elapsed = time.time() - t0

                if lc_err:
                    status.update(label="Download failed", state="error")
                    if "timeout" in lc_err.lower():
                        st.error(f"⏱️ MAST archive is busy — {lc_err} ({elapsed:.0f}s elapsed). Please retry in a moment.")
                    elif "not found" in lc_err.lower():
                        st.error(f"❌ {lc_err}")
                    else:
                        st.error(f"❌ {lc_err}")
                    st.stop()

                # Store in session so same-session reruns are instant
                st.session_state[f"lc_{kic_id}"] = (lc, n_quarters, n_cadences)
                st.write(f"✅ {n_quarters} quarters — {n_cadences:,} cadences "
                         f"in {elapsed:.1f}s")

            # ── Preprocess ──
            st.write("⚙️ Running preprocessing pipeline …")
            try:
                segment, flux_full, time_full, seg_start = preprocess(lc)
            except ValueError as ve:
                status.update(label="Preprocessing failed", state="error")
                st.error(f"❌ {ve}")
                st.stop()
            st.write("✅ Segment extracted and normalised")

            # ── Predict ──
            st.write("🧠 Running CNN inference …")
            x = segment.reshape(1, INPUT_LEN, 1)
            score = float(model.predict(x, verbose=0)[0][0])
            is_planet = score >= THRESHOLD
            tier, tier_cls, tier_label = get_confidence_tier(score)
            st.write(f"✅ Score: {score:.4f}")
            status.update(label="Pipeline complete", state="complete")

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # ── Result Box ──
        if is_planet:
            box_cls = "result-planet planet-pulse"
            emoji, headline = "🪐", "PLANET CANDIDATE DETECTED"
            sub = f"KIC {kic_id} shows a transit-like signal above the detection threshold."
        else:
            box_cls = "result-none"
            emoji, headline = "⭐", "NO PLANET DETECTED"
            sub = f"KIC {kic_id} does not exceed the detection threshold of {THRESHOLD}."

        st.markdown(f"""
        <div class='{box_cls}'>
          <div class='big-emoji'>{emoji}</div>
          <div class='result-title'>{headline}</div>
          <div style='color:#6a8db0; font-size:.85rem; margin:.4rem 0 .8rem;'>{sub}</div>
          <div class='conf-score'>{score:.4f}</div>
          <div style='font-size:.75rem; color:#4a6a8a; margin-bottom:.6rem;'>
            CNN CONFIDENCE SCORE &nbsp;|&nbsp; THRESHOLD {THRESHOLD}
          </div>
          <span class='tier-badge {tier_cls}'>{tier_label}</span>
        </div>""", unsafe_allow_html=True)

        # ── Low-score contextual warning ──
        if not is_planet:
            if score < 0.05:
                st.markdown("""
                <div class='warn-box'>
                  ⚠️ <strong>Not detected?</strong> Score near zero suggests the
                  best-variance segment landed on a starspot-dominated or flat region —
                  not a transit window. Possible causes: stellar variability (active star),
                  long orbital period (sparse transits in segment), or M-dwarf host
                  (out-of-distribution star type). See <em>Known Limitations</em> in
                  the Performance page for details.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='warn-box'>
                  ⚠️ <strong>Not detected?</strong> Score is above 0.05 but below
                  the 0.6914 threshold — a marginal signal. This may be a low-SNR planet,
                  a non-standard transit shape (e.g. grazing), or a borderline case.
                  Try the EB check and visual inspection of the light curve below.
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

        # ── EB Check ──
        eb_result = None
        if period_input > 0:
            with st.spinner(f"Running EB check at period {period_input:.4f} d …"):
                eb_result = run_eb_check(lc, period_input)

            if eb_result["skipped"]:
                # Skipped = no credible primary found at this period
                # This is NORMAL for a genuine planet — the phase fold
                # finds no eclipse deep enough to be real (below 5σ noise floor)
                if is_planet:
                    # Planet detected + EB skipped = good outcome
                    st.markdown(f"""
                    <div class='info-box'>
                      ✅ <strong>EB check: no secondary eclipse found</strong>
                      — phase fold at {period_input:.4f} d found no credible primary
                      eclipse above the 5σ noise floor. This is consistent with a
                      genuine planet transit (shallow, single-dip signature).<br>
                      <span style='font-size:.75rem; color:#3d6080;'>
                        Technical: {eb_result['reason']}
                      </span>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='warn-box'>
                      ⚠️ <strong>EB check inconclusive</strong>
                      — no primary eclipse detected above the 5σ noise floor at
                      period {period_input:.4f} d. Possible reasons: wrong period,
                      low SNR, or stellar variability masking the fold.<br>
                      <span style='font-size:.75rem; color:#806030;'>
                        Technical: {eb_result['reason']}
                      </span>
                    </div>""", unsafe_allow_html=True)

            elif eb_result["is_eb"]:
                st.markdown(f"""
                <div class='result-eb'>
                  <div class='big-emoji'>🚨</div>
                  <div class='result-title'>ECLIPSING BINARY FLAGGED</div>
                  <div style='color:#c06060; font-size:.85rem; margin:.4rem 0;'>
                    Secondary eclipse depth is {eb_result['ratio']:.1%} of primary —
                    exceeds the {EB_RATIO_THRESH:.0%} EB threshold.
                    This system is likely an eclipsing binary, not a planet host.
                  </div>
                  <div style='font-family:"Share Tech Mono",monospace; font-size:.88rem;
                              color:#e07070; margin-top:.6rem;
                              display:flex; gap:1.5rem; justify-content:center;'>
                    <span>Primary: {eb_result['primary_depth']:.5f}</span>
                    <span>Secondary: {eb_result['secondary_depth']:.5f}</span>
                    <span>Ratio: {eb_result['ratio']:.3f}</span>
                  </div>
                  <div style='font-size:.7rem; color:#6a4040; margin-top:.5rem;'>
                    Method: {eb_result['method']}
                  </div>
                </div>""", unsafe_allow_html=True)

            else:
                # Passed — secondary either absent or below threshold
                pri  = f"{eb_result['primary_depth']:.5f}" if eb_result['primary_depth'] else "—"
                sec  = f"{eb_result['secondary_depth']:.5f}" if eb_result['secondary_depth'] else "—"
                rat  = f"{eb_result['ratio']:.3f}" if eb_result['ratio'] is not None else "—"
                meth = eb_result['method'] or "—"
                st.markdown(f"""
                <div class='info-box'>
                  ✅ <strong>EB check passed</strong> — no significant secondary
                  eclipse detected at period {period_input:.4f} d.<br>
                  <span style='font-family:"Share Tech Mono",monospace; font-size:.8rem;
                               color:#4a8fb0;'>
                    Primary depth: {pri} &nbsp;|&nbsp;
                    Secondary: {sec} &nbsp;|&nbsp;
                    Ratio: {rat} &nbsp;|&nbsp;
                    Method: {meth}
                  </span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='info-box'>
              ℹ️ <strong>EB check not run</strong> — set a period above 0 to
              enable the eclipsing-binary rejection step. Period is auto-filled
              for the four demo stars.
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # ── Light Curve Plot ──
        st.markdown("""
        <h3 style='font-family:Orbitron,monospace; font-size:.95rem; letter-spacing:.1em;
                   color:#4a80b0;'>LIGHT CURVE</h3>""", unsafe_allow_html=True)

        seg_end = seg_start + INPUT_LEN
        time_seg = time_full[seg_start:seg_end]

        # Detect dip regions in the processed segment (below –2σ)
        seg_norm = segment - segment.mean()
        sigma    = seg_norm.std()
        dip_mask = seg_norm < (-2 * sigma)

        fig_lc = go.Figure()

        # Full light curve (faint)
        fig_lc.add_trace(go.Scatter(
            x=time_full, y=flux_full,
            mode="lines", name="Full LC",
            line=dict(color="rgba(42,100,180,0.25)", width=0.8),
            showlegend=True,
        ))

        # Selected segment highlight
        fig_lc.add_vrect(
            x0=time_seg[0], x1=time_seg[-1],
            fillcolor="rgba(37,117,252,0.07)",
            layer="below", line_width=0,
        )

        # Processed segment (scaled back for display)
        fig_lc.add_trace(go.Scatter(
            x=time_seg, y=segment * np.std(flux_full[seg_start:seg_end]) + np.mean(flux_full[seg_start:seg_end]),
            mode="lines", name="CNN input segment",
            line=dict(color="#2575fc", width=1.3),
        ))

        # Dip markers
        dip_times = time_seg[dip_mask]
        if len(dip_times) > 0:
            dip_flux = flux_full[seg_start:seg_end][dip_mask]
            fig_lc.add_trace(go.Scatter(
                x=dip_times, y=dip_flux,
                mode="markers", name="Flagged dips (–2σ)",
                marker=dict(color="#e74c3c", size=5, symbol="circle"),
            ))

        fig_lc.update_layout(
            **PLOT_LAYOUT,
            title=dict(text=f"KIC {kic_id} — Kepler PDCSAP Flux",
                       font=dict(family="Orbitron", color="#7ab3ff", size=14)),
            xaxis_title="Time (BKJD)",
            yaxis_title="Flux (e⁻/s)",
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_lc, use_container_width=True)

        # ── Preprocessed segment plot ──
        fig_seg = go.Figure()
        fig_seg.add_trace(go.Scatter(
            x=list(range(INPUT_LEN)), y=segment,
            mode="lines", name="Preprocessed",
            line=dict(color="#6a11cb", width=1.2),
        ))
        # shade dips
        fig_seg.add_trace(go.Scatter(
            x=list(np.where(dip_mask)[0]),
            y=segment[dip_mask],
            mode="markers", name="Flagged dips",
            marker=dict(color="#e74c3c", size=4),
        ))
        fig_seg.update_layout(
            **PLOT_LAYOUT,
            title=dict(text="Preprocessed segment fed to CNN",
                       font=dict(family="Orbitron", color="#9a70e0", size=13)),
            xaxis_title="Cadence index",
            yaxis_title="Normalised flux",
            height=280,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_seg, use_container_width=True)

        # ── Tier explanation ──
        st.markdown("""
        <h3 style='font-family:Orbitron,monospace; font-size:.9rem; letter-spacing:.1em;
                   color:#4a80b0; margin-top:1rem;'>CONFIDENCE TIER GUIDE</h3>""",
        unsafe_allow_html=True)

        tier_rows = [
            ("HIGH",   "> 0.50",   "tier-high", "Planet candidate — report and follow up with RV confirmation."),
            ("MEDIUM", "0.05–0.50","tier-med",  "Marginal signal — period fold + visual check recommended."),
            ("LOW",    "< 0.05",   "tier-low",  "Non-planet — discard. Likely flat or variable star."),
        ]
        for t, rng, cls, desc in tier_rows:
            active = "border:2px solid #2575fc;" if t == tier else ""
            st.markdown(f"""
            <div style='display:flex; gap:1rem; align-items:center; padding:.5rem 0;
                        border-bottom:1px solid #0a1929;'>
              <span class='tier-badge {cls}' style='{active} min-width:80px;
                    text-align:center;'>{t}</span>
              <span style='font-family:"Share Tech Mono",monospace; color:#2575fc;
                           font-size:.8rem; min-width:80px;'>{rng}</span>
              <span style='font-size:.82rem; color:#6a8db0;'>{desc}</span>
            </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════
elif page == "Performance":
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

    st.markdown("""
    <h1 style='font-family:Orbitron,monospace; font-size:1.6rem; font-weight:900;
               color:#7ab3ff; letter-spacing:.1em; margin-bottom:.3rem;'>
      📊 FINAL MODEL PERFORMANCE
    </h1>
    <div style='color:#4a6a8a; font-size:.85rem; margin-bottom:1.5rem;'>
      Benchmark discrimination, real-world evaluation, calibrated operating point, and scientific limitations.
    </div>""", unsafe_allow_html=True)

    # Headline metrics are reported evaluation results, not generated sample data.
    k1, k2, k3 = st.columns(3)
    for col, label, value, note in [
        (k1, "BENCHMARK ROC-AUC", "0.9628", "Competition test set"),
        (k2, "WILD-DATA ROC-AUC", "0.6933", "Real-world Kepler evaluation"),
        (k3, "DECISION THRESHOLD", "0.6914", "Configured operating point"),
    ]:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
              <div style='font-size:.68rem;color:#4a80b0;letter-spacing:.1em;margin-bottom:.5rem;'>{label}</div>
              <div style='font-family:Orbitron,monospace;font-size:1.65rem;color:#7ab3ff;font-weight:900;'>{value}</div>
              <div style='font-size:.75rem;color:#4a6a8a;margin-top:.35rem;'>{note}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <h3 style='font-family:Orbitron,monospace;font-size:.95rem;letter-spacing:.12em;color:#4a80b0;'>
      PERFORMANCE AT A GLANCE
    </h3>""", unsafe_allow_html=True)

    metric_left, metric_right = st.columns(2)
    for col, rel_path, caption in [
        (metric_left, "assets/performance_auc.png", "ROC-AUC across the benchmark and wild-data evaluations"),
        (metric_right, "assets/calibrated_metrics.png", "Calibrated operating point: precision, recall, F1, and false-positive rate"),
    ]:
        image_path = os.path.join(APP_DIR, rel_path)
        with col:
            if os.path.isfile(image_path):
                display_embedded_image(image_path, caption)
            else:
                st.warning(f"Performance chart not found: {rel_path}")

    st.markdown("""
    <h3 style='font-family:Orbitron,monospace;font-size:.95rem;letter-spacing:.12em;color:#4a80b0;margin-top:1rem;'>
      REPORTED OPERATING-POINT METRICS
    </h3>""", unsafe_allow_html=True)
    st.caption("The values below describe the reported calibrated evaluation sample (n=105). They should not be interpreted as universal performance on every Kepler target.")

    metrics_df = pd.DataFrame([
        {"Metric": "Precision", "Reported value": "1.000", "Interpretation": "No false positives at this evaluated operating point"},
        {"Metric": "Recall", "Reported value": "0.433", "Interpretation": "Some positive examples were missed"},
        {"Metric": "F1 score", "Reported value": "0.605", "Interpretation": "Balances precision and recall"},
        {"Metric": "False-positive rate", "Reported value": "0.000", "Interpretation": "Observed on this evaluation sample and filter configuration"},
        {"Metric": "Eclipsing-binary check", "Reported value": "6/6 flagged", "Interpretation": "Small tested sample; not a guarantee for all binaries"},
    ])
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    st.markdown("""
    <h3 style='font-family:Orbitron,monospace;font-size:.95rem;letter-spacing:.12em;color:#4a80b0;margin-top:1.25rem;'>
      SCIENTIFIC VISUALIZATIONS
    </h3>""", unsafe_allow_html=True)
    st.caption("Selected examples from the evaluation assets. These are curated figures, not a chronological development gallery.")

    visualizations = [
        ("results/week8/pr_curve.png", "Precision-recall trade-off on wild-data evaluation"),
        ("results/week8/eb_filter_results.png", "Eclipsing-binary filter evaluation"),
        ("results/week9/waveform_comparison.png", "Raw and processed stellar light-curve examples"),
        ("results/week9/lc_comparison.png", "Representative light-curve comparisons"),
    ]
    available_visualizations = []
    for rel_path, caption in visualizations:
        image_path = os.path.join(APP_DIR, rel_path)
        if os.path.isfile(image_path):
            available_visualizations.append((image_path, caption))

    if available_visualizations:
        for row_start in range(0, len(available_visualizations), 2):
            cols = st.columns(2)
            for col, (image_path, caption) in zip(cols, available_visualizations[row_start:row_start + 2]):
                with col:
                    display_embedded_image(image_path, caption)
    else:
        st.info("Evaluation figures are not available in this deployment. Check that the assets and results folders are committed to GitHub.")

    st.markdown("""
    <h3 style='font-family:Orbitron,monospace;font-size:.95rem;letter-spacing:.12em;color:#4a80b0;margin-top:1.25rem;'>
      LIMITATIONS AND INTERPRETATION
    </h3>""", unsafe_allow_html=True)
    limitations = [
        ("Benchmark-to-real-data gap", "ROC-AUC decreases from 0.9628 on the competition test set to 0.6933 on wild Kepler data."),
        ("Broader-catalog evaluation", "The saved evaluation reported 0% detection on a sample of 100 known planet hosts, showing that the current pipeline does not generalise reliably to that catalog."),
        ("Low signal-to-noise transits", "Shallow or noisy transit signals can be missed by the current model."),
        ("Stellar variability and instrumental effects", "Non-planetary changes in brightness can resemble or obscure transit signals."),
        ("Eclipsing binaries", "The screening step flags the tested examples, but a small test set does not establish general reliability."),
        ("Candidate scores are not confirmation", "A high CNN score is a screening result, not proof that an exoplanet exists. Independent follow-up is required."),
    ]
    for title, detail in limitations:
        st.markdown(f"""
        <div style='padding:.65rem 0;border-bottom:1px solid #0a1929;'>
          <div style='font-size:.85rem;color:#8aafd4;font-weight:600;'>{title}</div>
          <div style='font-size:.77rem;color:#4a6a8a;margin-top:.2rem;line-height:1.6;'>{detail}</div>
        </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ═════════════════════════════════════════════════════════════════
elif page == "About":
    st.markdown("""
    <h1 style='font-family:Orbitron,monospace; font-size:1.6rem; font-weight:900;
               color:#7ab3ff; letter-spacing:.1em; margin-bottom:.3rem;'>
      🪐 EXO HUNTER AI
    </h1>
    <div style='color:#4a6a8a; font-size:.85rem; margin-bottom:1.5rem;'>
      Deep Learning for Exoplanet Transit Detection
    </div>
    <div class='info-box' style='font-size:.9rem; line-height:1.8;'>
      Exo Hunter AI is an interactive astronomy and machine-learning project that
      analyzes stellar light curves from NASA's Kepler mission to identify patterns
      that may resemble an exoplanet transit. It brings together astronomical data
      retrieval, time-series preprocessing, a trained 1D Convolutional Neural Network
      (CNN), candidate scoring, and interactive scientific visualizations.
      <br><br>
      When a planet passes in front of its host star, the observed brightness can dip
      slightly. The application processes brightness measurements over time and uses
      the model to score transit-like signals for further inspection. A model result
      is a candidate signal, not confirmation that a planet exists; stellar variability,
      eclipsing binaries, instrument effects, and noise can produce similar patterns.
    </div>
    """, unsafe_allow_html=True)

    col_overview, col_stack = st.columns([1.1, 1])

    with col_overview:
        st.markdown("""
        <h3 style='font-family:Orbitron,monospace; font-size:.95rem; letter-spacing:.12em;
                   color:#4a80b0;'>PROJECT OVERVIEW</h3>
        """, unsafe_allow_html=True)

        overview = [
            ("01", "Data retrieval",
             "Uses Lightkurve and the Mikulski Archive for Space Telescopes (MAST) to retrieve Kepler observations for a selected KIC target."),
            ("02", "Light-curve preprocessing",
             "Prepares observed flux measurements for inference through the project's existing preprocessing pipeline, including handling missing values and normalization."),
            ("03", "CNN inference",
             "Passes the processed time series to a trained 1D CNN, which returns a score used by the application to flag transit-like candidates."),
            ("04", "Candidate checks",
             "Applies the configured decision threshold and an eclipsing-binary screening step. These checks can help flag possible false positives but do not replace scientific confirmation."),
            ("05", "Interactive analysis",
             "Visualizes the observed and processed light curve, marked signal regions, prediction score, and available model-evaluation results."),
        ]
        for number, title, desc in overview:
            st.markdown(f"""
            <div style='display:flex; gap:.9rem; align-items:flex-start; padding:.75rem 0;
                        border-bottom:1px solid #0a1929;'>
              <div style='font-family:Orbitron,monospace; font-size:.8rem; color:#2575fc;
                          min-width:28px; padding-top:.1rem;'>{number}</div>
              <div>
                <div style='font-size:.85rem; color:#8aafd4; font-weight:600;'>{title}</div>
                <div style='font-size:.76rem; color:#4a6a8a; line-height:1.6; margin-top:.2rem;'>{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <h3 style='font-family:Orbitron,monospace; font-size:.95rem; letter-spacing:.12em;
                   color:#4a80b0; margin-top:1.5rem;'>RESEARCH FOCUS</h3>
        <div style='font-size:.8rem; color:#4a6a8a; line-height:1.8;'>
          • Understand how deep learning can classify transit-like patterns in noisy stellar time series.<br>
          • Examine false positives, class imbalance, threshold choice, and model limitations.<br>
          • Compare performance claims only when the dataset and evaluation method are clearly identified.<br>
          • Treat predictions as candidates for follow-up, not as confirmed planetary discoveries.
        </div>
        """, unsafe_allow_html=True)

    with col_stack:
        st.markdown("""
        <h3 style='font-family:Orbitron,monospace; font-size:.95rem; letter-spacing:.12em;
                   color:#4a80b0;'>TECHNOLOGY</h3>
        """, unsafe_allow_html=True)

        stack = [
            ("🧠", "TensorFlow / Keras", "Loads and runs the existing 1D CNN model"),
            ("🌌", "Lightkurve + MAST", "Retrieves Kepler light-curve observations"),
            ("🔢", "NumPy + SciPy", "Numerical processing and signal preparation"),
            ("🐼", "Pandas", "Tabular data and catalogue handling"),
            ("📊", "Plotly", "Interactive light-curve and model-result charts"),
            ("🌐", "Streamlit", "Interactive web application"),
            ("🐍", "Python", "Application and ML pipeline implementation"),
        ]
        for icon, name, desc in stack:
            st.markdown(f"""
            <div style='display:flex; gap:.9rem; align-items:center; padding:.55rem 0;
                        border-bottom:1px solid #0a1929;'>
              <span style='font-size:1.2rem; min-width:28px; text-align:center;'>{icon}</span>
              <div>
                <div style='font-family:"Share Tech Mono",monospace; font-size:.82rem;
                            color:#7ab3ff;'>{name}</div>
                <div style='font-size:.75rem; color:#4a6a8a;'>{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <h3 style='font-family:Orbitron,monospace; font-size:.95rem; letter-spacing:.12em;
                   color:#4a80b0; margin-top:1.5rem;'>DATA SOURCES</h3>
        <div style='padding:.45rem 0; border-bottom:1px solid #0a1929;'>
          <div style='font-size:.82rem; color:#8aafd4;'>NASA Kepler mission</div>
          <div style='font-size:.73rem; color:#4a6a8a;'>Stellar photometry and light curves</div>
        </div>
        <div style='padding:.45rem 0; border-bottom:1px solid #0a1929;'>
          <div style='font-size:.82rem; color:#8aafd4;'>MAST archive</div>
          <div style='font-size:.73rem; color:#4a6a8a;'>Mission observations and data products</div>
        </div>
        <div style='padding:.45rem 0; border-bottom:1px solid #0a1929;'>
          <div style='font-size:.82rem; color:#8aafd4;'>NASA Exoplanet Archive</div>
          <div style='font-size:.73rem; color:#4a6a8a;'>Catalogue context and exoplanet reference data</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <h3 style='font-family:Orbitron,monospace; font-size:.95rem; letter-spacing:.12em;
                   color:#4a80b0; margin-top:1.5rem;'>PROJECT OWNER</h3>
        <div style='font-size:.82rem; color:#8aafd4; margin-bottom:.35rem;'>Krish Gupta</div>
        <div style='font-size:.76rem; color:#4a6a8a; line-height:1.6;'>
          Portfolio project focused on astronomy, machine learning, and interactive scientific applications.
        </div>
        <div style='margin-top:.8rem;'>
          <a href='https://github.com/krishgupta129' target='_blank'
             style='display:inline-block; padding:.55rem .85rem; border:1px solid #2575fc;
                    border-radius:7px; color:#7ab3ff; text-decoration:none; font-size:.8rem;'>
            ↗ Explore my GitHub projects
          </a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <h3 style='font-family:Orbitron,monospace; font-size:.95rem; letter-spacing:.12em;
               color:#4a80b0;'>MODEL SUMMARY</h3>
    """, unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    model_card = [
        ("Architecture", "1D Convolutional Neural Network"),
        ("Input", "Processed stellar light curve"),
        ("Output", "Transit-like signal score"),
        ("Decision", "Configured classification threshold"),
    ]
    for col, (k, v) in zip([mc1, mc2, mc3, mc4], model_card):
        col.markdown(f"""
        <div class='metric-card' style='padding:.9rem 1rem;'>
          <div style='font-size:.65rem; color:#2d4d6e; letter-spacing:.1em;
                      margin-bottom:.3rem;'>{k}</div>
          <div style='font-family:"Share Tech Mono",monospace; font-size:.82rem;
                      color:#7ab3ff;'>{v}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box' style='font-size:.78rem; line-height:1.7;'>
      <b>Scientific note:</b> A high model score is not proof that an exoplanet exists.
      Performance metrics depend on the dataset and evaluation protocol; broader-catalog
      results may differ from curated benchmark results. Candidates require independent
      scientific validation.<br><br>
      <b>Project and attribution:</b> This portfolio version is maintained by Krish Gupta.
      It builds on an MIT-licensed codebase by Mann Bhatt; the original license and copyright
      notice are retained in the repository.
    </div>
    """, unsafe_allow_html=True)

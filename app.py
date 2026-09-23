from pathlib import Path
from datetime import datetime, timezone
import json
import io

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Realtime Seizure Monitoring",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL_PATH = OUT / "best_model.pkl"
SCALER_PATH = OUT / "scaler.pkl"
META_PATH = OUT / "metadata.joblib"
BUNDLE_PATH = OUT / "best_model_bundle.joblib"
DATA_DIR = ROOT / "data"
LOCAL_DATA_PATH = DATA_DIR / "epileptic_seizure_data.csv"
DATA_URL = "https://raw.githubusercontent.com/Jreevo/Epileptic-Seizure-Binary-Classification/master/epilepsy.csv"

st.markdown(
    """
<style>
:root{--bg:#071218;--panel:#0b1b22;--panel2:#10262d;--line:#21414a;--text:#f2fbfa;--muted:#8ea9ad;--teal:#35e0c2;--cyan:#56d9ff;--mint:#67e8b3;--rose:#ff6b7a;--amber:#ffca6b;--blue:#7eb6ff;--purple:#9b8cff}
.stApp{background:radial-gradient(circle at 85% -10%,rgba(53,224,194,.18),transparent 34%),radial-gradient(circle at 10% 0%,rgba(86,217,255,.12),transparent 28%),linear-gradient(135deg,#071218 0%,#08161d 52%,#061017 100%);color:var(--text)}
.block-container{max-width:1540px;padding:.25rem 1.7rem 2.2rem}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0a0f1d,#080c16);border-right:1px solid #1d2940}
.hero{display:flex;justify-content:space-between;align-items:center;gap:20px;padding:26px 29px;border:1px solid #293552;border-radius:24px;background:linear-gradient(135deg,rgba(17,25,44,.97),rgba(10,14,27,.98));box-shadow:0 22px 70px rgba(0,0,0,.30)}
.hero h1{margin:0;font-size:2.65rem;letter-spacing:-.055em;background:linear-gradient(90deg,#fff,var(--cyan),var(--teal));-webkit-background-clip:text;background-clip:text;color:transparent}.hero p{margin:.45rem 0 0;color:#9aa8bf}.kicker{font-size:.7rem;letter-spacing:.16em;font-weight:850;color:var(--cyan)}
.live-pill{display:flex;align-items:center;gap:9px;padding:10px 15px;border-radius:999px;border:1px solid #344264;background:rgba(21,31,53,.8);font-weight:850;font-size:.8rem;white-space:nowrap}.live-dot{width:9px;height:9px;border-radius:50%;background:var(--mint);box-shadow:0 0 0 0 rgba(78,225,181,.55);animation:pulse 1.45s infinite}
@keyframes pulse{70%{box-shadow:0 0 0 9px rgba(78,225,181,0)}100%{box-shadow:0 0 0 0 rgba(78,225,181,0)}}
.card{background:linear-gradient(145deg,rgba(13,31,38,.97),rgba(7,19,25,.99));border:1px solid var(--line);border-radius:18px;padding:16px 17px;box-shadow:0 10px 35px rgba(0,0,0,.12)}
.metric{font-size:1.7rem;font-weight:850;line-height:1.1}.label{font-size:.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em}.sub{font-size:.76rem;color:#9ba9bd;margin-top:6px}.small-muted{color:var(--muted);font-size:.74rem}
.alert-banner,.safe-banner{border-radius:18px;padding:14px 18px;margin:12px 0;border:1px solid}.alert-banner{border-color:rgba(255,102,133,.58);background:linear-gradient(90deg,rgba(255,102,133,.13),rgba(139,124,255,.06));animation:alertFlash 1s ease-in-out infinite alternate}.safe-banner{border-color:rgba(78,225,181,.32);background:linear-gradient(90deg,rgba(78,225,181,.075),rgba(79,215,255,.035))}
@keyframes alertFlash{from{box-shadow:0 0 0 rgba(255,102,133,0)}to{box-shadow:0 0 34px rgba(255,102,133,.17)}}
.section-title{font-size:1rem;font-weight:850;margin:4px 0 12px}.section-title span{color:var(--muted);font-size:.78rem;font-weight:500;margin-left:7px}.session-tag{display:inline-block;padding:5px 9px;border-radius:9px;background:#151f36;color:#b9c5dc;font-size:.7rem;margin-right:5px;border:1px solid #2b3958}
.top-telemetry{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:12px 0}.telemetry{background:rgba(8,25,32,.88);border:1px solid #1c3c45;border-radius:15px;padding:10px 13px;position:relative;overflow:hidden}.telemetry:after{content:"";position:absolute;top:0;bottom:0;width:80px;background:linear-gradient(90deg,transparent,rgba(53,224,194,.10),transparent);animation:sweep 3s linear infinite}.telemetry .k{font-size:.61rem;color:#7786a0;text-transform:uppercase;letter-spacing:.11em}.telemetry .v{font-size:1.02rem;font-weight:850;margin-top:2px}.telemetry .s{font-size:.67rem;color:#93a0b6;margin-top:2px}.live-text{color:var(--mint)}.standby-text{color:#9aa7ba}.safe-text{color:var(--mint)}.alert-text{color:var(--rose)}@keyframes sweep{from{left:-100px}to{left:100%}}
.monitor-shell{border:1px solid #293753;border-radius:21px;background:linear-gradient(145deg,#0a101d,#0d1525);padding:14px;box-shadow:inset 0 1px 0 rgba(255,255,255,.035),0 20px 60px rgba(0,0,0,.20)}
.monitor-head{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:9px}.monitor-head .title{font-size:1.05rem;font-weight:850}.monitor-head .meta{font-size:.71rem;color:#7786a0;margin-top:2px}
.pulse-line{height:3px;border-radius:4px;background:linear-gradient(90deg,transparent,var(--cyan),var(--teal),transparent);background-size:220% 100%;animation:scan 1.25s linear infinite;opacity:.85}@keyframes scan{to{background-position:-220% 0}}
.alert-dot,.safe-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}.alert-dot{background:var(--rose);box-shadow:0 0 0 0 rgba(255,102,133,.5);animation:pulse 1s infinite}.safe-dot{background:var(--mint);box-shadow:0 0 0 0 rgba(78,225,181,.5);animation:pulse 1.7s infinite}
.stButton>button{border-radius:12px;border:1px solid #2b3a59;background:linear-gradient(180deg,#112a31,#0c2027);color:#f4f7ff;font-weight:800;min-height:43px;transition:.18s}.stButton>button:hover{border-color:var(--teal);background:linear-gradient(180deg,#123039,#0d252d);transform:translateY(-1px)}
.stDownloadButton>button{border-radius:12px!important;border:1px solid #2b3a59!important;background:#111a2d!important;color:#f4f7ff!important}
div[data-baseweb="select"]>div, .stTextInput input, .stTextArea textarea{background:#0b2027!important;border-color:#28505a!important;color:#f5f7ff!important;border-radius:11px!important}
[data-testid="stMetric"]{background:transparent}/* Single dark clinical mode */
html,body,[data-testid="stAppViewContainer"],[data-testid="stHeader"]{background:#071218!important;color:var(--text)!important}
footer{visibility:hidden}
/* True single dark mode: uploader and native controls */
section[data-testid="stFileUploaderDropzone"]{background:#0b2027!important;border:1px solid #28505a!important;border-radius:12px!important}
section[data-testid="stFileUploaderDropzone"] *{color:#dff8f5!important}
section[data-testid="stFileUploaderDropzone"] svg{fill:#67e8d0!important;color:#67e8d0!important}
section[data-testid="stFileUploaderDropzone"] button{background:#102d35!important;color:#f2fbfa!important;border:1px solid #3b6872!important}
section[data-testid="stFileUploaderDropzone"] button:hover{background:#153b44!important;border-color:#35e0c2!important}
[data-testid="stFileUploader"] small{color:#8ea9ad!important}
[data-testid="stFileUploader"] label{color:#dff8f5!important}
.stFileUploader{color:#dff8f5!important}
/* Compact full-width current EEG trace with no artificial vertical gap */
.live-current-trace{width:100%!important;margin-top:0!important;padding-top:0!important;min-height:0!important}
.live-current-trace .stPlotlyChart{margin-top:0!important;margin-bottom:0!important}

/* Consistent breathing room between UI blocks */
.stMarkdown,.stButton,.stDownloadButton,.stFileUploader,.stSelectbox,.stTextInput,.stTextArea,.stNumberInput,.stDataFrame,.stPlotlyChart,.stAlert,.stTabs,.stSlider,.stRadio,.stCheckbox{margin-bottom:14px!important}
[data-testid="stVerticalBlock"] > div{margin-bottom:6px}
.card,.monitor-shell,.upload-shell,.analytics-hero,.analytics-grid,.top-telemetry,.alert-banner,.safe-banner{margin-bottom:18px!important}


.analytics-hero{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;padding:18px 20px;border:1px solid #24505a;border-radius:20px;background:linear-gradient(135deg,rgba(17,48,57,.92),rgba(8,24,30,.98));margin-bottom:14px}.analytics-hero h2{margin:0;font-size:1.45rem;letter-spacing:-.02em}.analytics-hero p{margin:5px 0 0;color:#9bb5b9;font-size:.8rem}.analytics-badge{padding:7px 11px;border-radius:999px;background:rgba(53,224,194,.10);border:1px solid rgba(53,224,194,.28);color:#75f0d5;font-size:.7rem;font-weight:800;white-space:nowrap}.analytics-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:12px 0 16px}.analytics-stat{padding:14px;border-radius:16px;background:linear-gradient(145deg,#0d252d,#091a21);border:1px solid #21454e}.analytics-stat .n{font-size:1.55rem;font-weight:900}.analytics-stat .l{font-size:.66rem;text-transform:uppercase;letter-spacing:.11em;color:#7f9ca1;margin-top:3px}.analytics-stat .s{font-size:.72rem;color:#9bb0b4;margin-top:5px}.upload-shell{padding:18px;border:1px solid #24505a;border-radius:20px;background:linear-gradient(145deg,rgba(13,35,43,.96),rgba(7,20,26,.99));box-shadow:0 14px 40px rgba(0,0,0,.16);margin:8px 0 14px}.upload-title{font-size:1.08rem;font-weight:900}.upload-sub{font-size:.76rem;color:#8fa9ad;margin-top:3px}.mini-chip{display:inline-block;padding:4px 8px;border-radius:8px;background:#103039;border:1px solid #24535d;color:#9cefe1;font-size:.67rem;margin-right:5px}.analytics-note{padding:10px 13px;border-left:3px solid var(--teal);background:rgba(53,224,194,.055);border-radius:8px;color:#9fb6b9;font-size:.75rem}
@media(max-width:1100px){.analytics-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:900px){.hero h1{font-size:1.8rem}.top-telemetry{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    meta = joblib.load(META_PATH) if META_PATH.exists() else {}
    if BUNDLE_PATH.exists():
        bundle = joblib.load(BUNDLE_PATH)
        return bundle.get("model"), bundle.get("scaler"), meta, float(bundle["threshold"]), bundle.get("model_name", "Unknown")
    if MODEL_PATH.exists() and SCALER_PATH.exists() and isinstance(meta, dict) and "alert_threshold" in meta:
        return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH), meta, float(meta["alert_threshold"]), meta.get("best_name", "Unknown")
    return None, None, meta, None, None


model, scaler, meta, trained_threshold, trained_model_name = load_artifacts()


def init_state():
    defaults = {
        "monitoring": False,
        "started_at": None,
        "tick": 0,
        "alerts": [],
        "last_prediction": 0,
        "last_probability": 0.0,
        "last_signal": None,
        "last_ts": None,
        "previous_prediction": 0,
        "csv_df": None,
        "csv_features": None,
        "csv_row": 0,
        "patient_id": "PT-001",
        "patient_name": "Research Session",
        "session_note": "EEG screening session",
        "threshold": None,
        "source": "Real seizure replay",
        "source_detail": "Real EEG replay dataset",
        "replay_filter": "Seizure",
        "replay_row": 0,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


init_state()

st.markdown(
    """
<div class="hero">
  <div>
    <div class="kicker">CLINICAL RESEARCH • EEG ANALYTICS • LIVE MONITOR</div>
    <h1>Realtime Seizure Monitoring</h1>
    <p>Hospital-style EEG command center for real EEG replay, live screening, event surveillance and session reporting.</p>
  </div>
  <div class="live-pill"><span class="live-dot"></span> MONITOR READY</div>
</div>
""",
    unsafe_allow_html=True,
)

if model is None or scaler is None or trained_threshold is None:
    st.error("Deployment artifacts are incomplete. Run the updated notebook export so the learned threshold is stored in outputs/best_model_bundle.joblib.")
    st.stop()


# Sidebar
st.sidebar.markdown("## 🏥 Control Room")
st.sidebar.caption("Realtime Seizure Monitoring")
patient_id = st.sidebar.text_input("Patient / Case ID", value=st.session_state.patient_id)
patient_name = st.sidebar.text_input("Session label", value=st.session_state.patient_name)
session_note = st.sidebar.text_area("Clinical / research note", value=st.session_state.session_note, height=80)
threshold = float(trained_threshold)
source_options = ["Real seizure replay", "Real non-seizure replay", "EEG CSV", "Single EEG window"]
source = st.sidebar.selectbox("Stream source", source_options, index=source_options.index(st.session_state.source) if st.session_state.source in source_options else 0)
st.session_state.update(patient_id=patient_id, patient_name=patient_name, session_note=session_note, threshold=threshold, source=source)

if st.sidebar.button("🧹 Clear alert history", use_container_width=True):
    st.session_state.alerts = []
    st.session_state.previous_prediction = 0

st.sidebar.markdown("---")
st.sidebar.markdown("**Learned alert threshold**")
st.sidebar.info(f"{threshold:.1%} · validation F1 optimized")
st.sidebar.caption("Read-only: loaded from the trained deployment bundle.")
st.sidebar.markdown("**Safety**")
st.sidebar.caption("Research/demo interface. Alerts are model outputs and must not be treated as a diagnosis or emergency medical decision.")

if source in ("Real seizure replay", "Real non-seizure replay"):
    st.sidebar.success("Using real EEG windows from the project dataset — not a synthetic waveform.")


def get_csv_signal():
    st.markdown(
        '<div class="upload-shell"><div class="upload-title">📁 EEG CSV workspace</div><div class="upload-sub">Upload one or more EEG windows. The app validates X1–X178, previews the selected window, and lets you inspect the signal before inference.</div></div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Drop your EEG CSV here",
        type=["csv"],
        help="CSV should contain X1–X178. A y/label column is allowed and will be ignored for model input.",
    )
    if uploaded is None:
        st.info("Waiting for a CSV file. Recommended format: X1, X2, …, X178, with one EEG window per row.")
        return None, None
    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        return None, None
    feature_cols = [f"X{i}" for i in range(1, 179)]
    if not all(c in df.columns for c in feature_cols):
        lower_map = {str(c).strip().lower(): c for c in df.columns}
        aliases = [lower_map.get(f"x{i}") for i in range(1,179)]
        if all(a is not None for a in aliases):
            feature_cols = aliases
        else:
            numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and str(c).lower() not in {"y","target","label"}]
            if len(numeric) == 178:
                feature_cols = numeric
    if len(feature_cols) != 178:
        st.error(f"Expected 178 EEG columns (X1–X178); detected {len(feature_cols)} usable columns.")
        st.caption("Tip: keep the feature names X1 through X178. Extra columns such as y, target, label, or an ID are fine.")
        return None, None
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    valid_mask = df[feature_cols].notna().all(axis=1)
    if not valid_mask.all():
        st.warning(f"{int((~valid_mask).sum())} row(s) contain missing/non-numeric EEG values. Those rows may not be suitable for inference.")
    st.session_state.csv_df = df
    st.session_state.csv_features = feature_cols
    max_row = max(len(df) - 1, 0)
    row = st.slider("Select EEG window", 0, max_row, min(st.session_state.csv_row, max_row), format="Window %d")
    st.session_state.csv_row = row
    vals = df[feature_cols].iloc[row].astype(float).values
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Rows / windows", f"{len(df):,}")
    with c2: st.metric("EEG features", "178")
    with c3: st.metric("Selected window", f"{row+1:,}")
    with c4: st.metric("Signal RMS", f"{np.sqrt(np.mean(np.square(vals))):.3f}")
    fig = go.Figure(go.Scatter(x=np.arange(1,179), y=vals, mode="lines", line=dict(color="#35e0c2", width=2), fill="tozeroy", fillcolor="rgba(53,224,194,.055)"))
    fig.update_layout(template="plotly_dark", height=260, margin=dict(l=5,r=5,t=18,b=5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#071b22", xaxis=dict(title="Feature sample", gridcolor="#17383f"), yaxis=dict(title="Amplitude", gridcolor="#17383f"), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo":False,"scrollZoom":False})
    st.markdown(f'<span class="mini-chip">CSV loaded</span><span class="mini-chip">178 features validated</span><span class="mini-chip">Window {row+1} ready</span>', unsafe_allow_html=True)
    return vals, f"CSV row {row + 1}/{len(df)}"


def parse_single():
    st.markdown(
        '<div class="upload-shell"><div class="upload-title">⌨️ Single EEG window</div><div class="upload-sub">Enter the 178 model features in a compact spreadsheet-style grid. Only the value cells are editable.</div></div>',
        unsafe_allow_html=True,
    )
    if "manual_values" not in st.session_state or len(st.session_state.manual_values) != 178:
        st.session_state.manual_values = [0.0] * 178
    rows = 36
    data = {}
    for pair in range(5):
        fcol, vcol = f"Feature {pair+1}", f"Value {pair+1}"
        features=[]; values=[]
        for r in range(rows):
            idx = r*5 + pair
            if idx < 178:
                features.append(f"X{idx+1}")
                values.append(float(st.session_state.manual_values[idx]))
            else:
                features.append("")
                values.append(np.nan)
        data[fcol]=features; data[vcol]=values
    grid = pd.DataFrame(data)
    edited = st.data_editor(
        grid,
        hide_index=True,
        use_container_width=True,
        height=610,
        disabled=["Feature 1","Feature 2","Feature 3","Feature 4","Feature 5"],
        column_config={
            "Feature 1": st.column_config.TextColumn("Feature", width="small"), "Value 1": st.column_config.NumberColumn("Value", format="%.6f", width="medium"),
            "Feature 2": st.column_config.TextColumn("Feature", width="small"), "Value 2": st.column_config.NumberColumn("Value", format="%.6f", width="medium"),
            "Feature 3": st.column_config.TextColumn("Feature", width="small"), "Value 3": st.column_config.NumberColumn("Value", format="%.6f", width="medium"),
            "Feature 4": st.column_config.TextColumn("Feature", width="small"), "Value 4": st.column_config.NumberColumn("Value", format="%.6f", width="medium"),
            "Feature 5": st.column_config.TextColumn("Feature", width="small"), "Value 5": st.column_config.NumberColumn("Value", format="%.6f", width="medium"),
        },
        key="manual_eeg_grid",
    )
    values = [0.0]*178
    for pair in range(5):
        for r in range(rows):
            idx = r*5+pair
            if idx < 178:
                val = edited.iloc[r][f"Value {pair+1}"]
                values[idx] = 0.0 if pd.isna(val) else float(val)
    st.session_state.manual_values = values
    pasted = st.text_area(
        "Optional: paste all 178 values at once",
        height=80,
        placeholder="Paste comma-, space-, or newline-separated values here…",
        help="If provided, click Apply pasted values to load them into the table above.",
        key="manual_paste",
    )
    if st.button("↳ Apply pasted values", use_container_width=True, key="apply_paste"):
        arr = np.fromstring(pasted.replace(","," ").replace("\n"," "), sep=" ") if pasted.strip() else np.array([])
        if len(arr) != 178:
            st.error(f"Expected exactly 178 values; received {len(arr)}.")
        else:
            st.session_state.manual_values = arr.astype(float).tolist()
            st.rerun()
    vals = np.asarray(st.session_state.manual_values, dtype=float)
    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Features entered", "178 / 178")
    with c2: st.metric("Signal RMS", f"{np.sqrt(np.mean(np.square(vals))):.3f}")
    with c3: st.metric("Peak amplitude", f"{np.max(np.abs(vals)):.3f}")
    fig = go.Figure(go.Scatter(x=np.arange(1,179), y=vals, mode="lines", line=dict(color="#56d9ff", width=2), fill="tozeroy", fillcolor="rgba(86,217,255,.05)"))
    fig.update_layout(template="plotly_dark", height=250, margin=dict(l=5,r=5,t=18,b=5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#071b22", xaxis=dict(title="Feature index", gridcolor="#17383f"), yaxis=dict(title="Value", gridcolor="#17383f"), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo":False,"scrollZoom":False})
    return vals, "Manual EEG window"


@st.cache_data(ttl=3600, show_spinner=False)
def load_replay_dataset():
    """Load the same 11,500-row EEG dataset used by the training notebook.

    Prefer a local copy in data/ for deployment stability; otherwise use the
    public GitHub mirror referenced by the notebook. Only X1-X178 and y are kept.
    """
    if LOCAL_DATA_PATH.exists():
        df = pd.read_csv(LOCAL_DATA_PATH)
    else:
        df = pd.read_csv(DATA_URL)
    feature_cols = [f"X{i}" for i in range(1, 179)]
    missing = [c for c in feature_cols + ["y"] if c not in df.columns]
    if missing:
        raise ValueError(f"Replay dataset is missing columns: {missing[:8]}")
    df = df[feature_cols + ["y"]].copy()
    df["binary_label"] = (pd.to_numeric(df["y"], errors="coerce") == 1).astype(int)
    return df


def get_replay_signal(mode):
    try:
        df = load_replay_dataset()
    except Exception as exc:
        st.error(f"Could not load the real EEG replay dataset: {exc}")
        st.info("For reliable deployment, place data/epileptic_seizure_data.csv in the GitHub repository. The notebook uses the same public dataset.")
        return None, None
    target = 1 if mode == "Real seizure replay" else 0
    subset = df[df["binary_label"] == target]
    if subset.empty:
        return None, None
    row = st.session_state.replay_row % len(subset)
    cols = [f"X{i}" for i in range(1, 179)]
    signal = subset.iloc[row][cols].astype(float).values
    label_text = "Seizure" if target else "Non-Seizure"
    return signal, f"Real EEG replay · {label_text} · sample {row + 1}/{len(subset)}"


if source == "EEG CSV":
    signal, source_detail = get_csv_signal()
elif source == "Single EEG window":
    signal, source_detail = parse_single()
else:
    signal, source_detail = get_replay_signal(source)


def predict(signal):
    x = np.asarray(signal, dtype=float).reshape(1, -1)
    if x.shape[1] != 178:
        raise ValueError(f"Model expects 178 features, received {x.shape[1]}.")
    scaled = scaler.transform(x)
    proba = float(model.predict_proba(scaled)[0, 1])
    pred = int(proba >= trained_threshold)
    return pred, proba


# Controls
c1, c2, c3, c4 = st.columns([1.05, 1.05, 1.05, 2.85])
with c1:
    if st.button("▶ Start monitoring", use_container_width=True, disabled=st.session_state.monitoring):
        st.session_state.monitoring = True
        st.session_state.started_at = datetime.now(timezone.utc)
        st.session_state.tick = 0
        st.session_state.replay_row = 0
        st.session_state.alerts = []
        st.session_state.previous_prediction = 0
        st.rerun()
with c2:
    if st.button("■ Stop", use_container_width=True, disabled=not st.session_state.monitoring):
        st.session_state.monitoring = False
        st.rerun()
with c3:
    if st.button("↻ Reset", use_container_width=True):
        st.session_state.monitoring = False
        st.session_state.tick = 0
        st.session_state.replay_row = 0
        st.session_state.alerts = []
        st.session_state.previous_prediction = 0
        st.session_state.last_signal = None
        st.session_state.last_probability = 0.0
        st.session_state.started_at = None
        st.rerun()
with c4:
    state_text = "LIVE STREAMING" if st.session_state.monitoring else "STANDBY"
    state_color = "#56df9b" if st.session_state.monitoring else "#8297a8"
    st.markdown(f'<div class="card" style="padding:11px 15px"><b style="color:{state_color}">● {state_text}</b><span class="small-muted"> &nbsp; {st.session_state.get("source_detail", source_detail or "No stream")}</span></div>', unsafe_allow_html=True)


@st.fragment(run_every="700ms")
def live_command_center():
    # Actual repeated model inference while monitoring is active.
    if st.session_state.monitoring:
        tick = st.session_state.tick
        if source in ("Real seizure replay", "Real non-seizure replay"):
            live_signal, detail = get_replay_signal(source)
            st.session_state.replay_row += 1
        elif source == "EEG CSV" and st.session_state.csv_df is not None:
            df = st.session_state.csv_df
            cols = st.session_state.csv_features
            row = st.session_state.csv_row % len(df)
            live_signal = df[cols].iloc[row].astype(float).values
            detail = f"CSV row {row + 1}/{len(df)}"
            st.session_state.csv_row = (row + 1) % len(df)
        else:
            live_signal = signal if signal is not None else np.zeros(178)
            detail = source_detail or "Manual EEG window"
        if live_signal is not None:
            try:
                pred, proba = predict(live_signal)
                now = datetime.now(timezone.utc)
                st.session_state.tick += 1
                st.session_state.last_prediction = pred
                st.session_state.last_probability = proba
                st.session_state.last_signal = live_signal
                st.session_state.last_ts = now
                st.session_state.source_detail = detail
                previous = st.session_state.previous_prediction
                if pred == 1 and previous == 0:
                    st.session_state.alerts.insert(0, {
                        "time": now.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
                        "patient": st.session_state.patient_id,
                        "probability": round(proba, 4),
                        "threshold": st.session_state.threshold,
                        "source": detail,
                    })
                    st.session_state.alerts = st.session_state.alerts[:50]
                st.session_state.previous_prediction = pred
            except Exception as exc:
                st.error(f"Live inference error: {exc}")

    if st.session_state.last_signal is None and signal is not None and not st.session_state.monitoring:
        try:
            p, pr = predict(signal)
            st.session_state.last_prediction = p
            st.session_state.last_probability = pr
            st.session_state.last_signal = signal
        except Exception:
            pass

    prob = float(st.session_state.last_probability)
    pred = int(st.session_state.last_prediction)
    elapsed = 0
    if st.session_state.started_at:
        elapsed = max(0, int((datetime.now(timezone.utc) - st.session_state.started_at).total_seconds()))
    status = "SEIZURE ALERT" if pred else "NO SEIZURE DETECTED"

    if pred:
        st.markdown(f'<div class="alert-banner"><b><span class="alert-dot"></span>{status}</b><br><span class="small-muted">Model probability {prob:.1%} is at/above the configured alert threshold of {threshold:.1%}. Review the waveform and event timeline.</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="safe-banner"><b><span class="safe-dot"></span>{status}</b><br><span class="small-muted">Current model probability {prob:.1%} is below the configured alert threshold of {threshold:.1%}.</span></div>', unsafe_allow_html=True)

    state_class = "live-text" if st.session_state.monitoring else "standby-text"
    latest_ts = st.session_state.last_ts.astimezone().strftime("%H:%M:%S") if st.session_state.last_ts else "--:--:--"
    st.markdown(f'''<div class="top-telemetry">
      <div class="telemetry"><div class="k">Monitor state</div><div class="v {state_class}">● {"LIVE" if st.session_state.monitoring else "STANDBY"}</div><div class="s">700 ms inference cadence</div></div>
      <div class="telemetry"><div class="k">Latest inference</div><div class="v">{latest_ts}</div><div class="s">{st.session_state.get("source_detail", source_detail or "No stream")}</div></div>
      <div class="telemetry"><div class="k">Window</div><div class="v">178 samples</div><div class="s">EEG feature vector X1–X178</div></div>
      <div class="telemetry"><div class="k">Learned alert threshold</div><div class="v">{threshold:.0%}</div><div class="s">Validation F1 optimized</div></div>
    </div>''', unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f'<div class="card"><div class="label">Patient / case</div><div class="metric">{st.session_state.patient_id}</div><div class="sub">{st.session_state.patient_name}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="card"><div class="label">Seizure probability</div><div class="metric">{prob:.1%}</div><div class="sub">Threshold {threshold:.0%}</div></div>', unsafe_allow_html=True)
    with m3:
        rms = float(np.sqrt(np.mean(np.square(np.asarray(st.session_state.last_signal, dtype=float))))) if st.session_state.last_signal is not None else 0.0
        st.markdown(f'<div class="card"><div class="label">EEG signal RMS</div><div class="metric">{rms:.1f}</div><div class="sub">Live window amplitude summary</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="card"><div class="label">Alert events</div><div class="metric">{len(st.session_state.alerts):02d}</div><div class="sub">This monitoring session</div></div>', unsafe_allow_html=True)
    with m5:
        st.markdown(f'<div class="card"><div class="label">Session time</div><div class="metric">{elapsed//60:02d}:{elapsed%60:02d}</div><div class="sub">{model.__class__.__name__}</div></div>', unsafe_allow_html=True)

    if source in ("Real seizure replay", "Real non-seizure replay"):
        replay_label = 1 if source == "Real seizure replay" else 0
        replay_text = "SEIZURE" if replay_label else "NON-SEIZURE"
        agreement = "MATCH" if pred == replay_label else "MISMATCH"
        agreement_cls = "safe-text" if pred == replay_label else "alert-text"
        st.markdown(f'<div class="card" style="margin:10px 0;border-color:#2b3958"><div class="label">Replay ground truth</div><div class="metric">{replay_text}</div><div class="sub">Model output: {"SEIZURE" if pred else "NON-SEIZURE"} · <b class="{agreement_cls}">{agreement}</b> · demonstration label comes from the dataset</div></div>', unsafe_allow_html=True)

    left, right = st.columns([2.15, 1])
    with left:
        st.markdown('<div class="monitor-shell"><div class="monitor-head"><div><div class="title">Live EEG waveform</div><div class="meta">Streaming neural signal · 178 samples · model input window</div></div><span class="session-tag">LIVE EEG</span></div>', unsafe_allow_html=True)
        if st.session_state.last_signal is not None:
            y = np.asarray(st.session_state.last_signal)
            x = np.arange(1, len(y) + 1)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="#4fd7ff", width=2.3), fill="tozeroy", fillcolor="rgba(79,215,255,.045)", hovertemplate="Sample %{x}<br>Amplitude %{y:.4f}<extra></extra>"))
            fig.update_layout(template="plotly_dark", height=400, margin=dict(l=5,r=5,t=8,b=5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#06131b", xaxis=dict(title="Sample", gridcolor="#173040", zeroline=False), yaxis=dict(title="EEG amplitude", gridcolor="#173040", zeroline=False), showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": False})
        else:
            st.info("Load an EEG source and start monitoring to populate the live waveform.")
    with right:
        st.markdown('<div class="section-title">Signal status <span>model output</span></div>', unsafe_allow_html=True)
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=prob*100, number={"suffix":"%", "font":{"size":38}}, title={"text":"Seizure probability"}, gauge={"axis":{"range":[0,100]}, "bar":{"color":"#ff5d73" if pred else "#56df9b", "thickness":.28}, "steps":[{"range":[0,threshold*100],"color":"rgba(86,223,155,.10)"},{"range":[threshold*100,100],"color":"rgba(255,93,115,.10)"}], "threshold":{"line":{"color":"#ffc45c","width":4},"value":threshold*100}}))
        gauge.update_layout(template="plotly_dark", height=270, margin=dict(l=10,r=10,t=25,b=5), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(gauge, use_container_width=True, config={"displaylogo": False})
        st.markdown(f'<div class="card"><span class="session-tag">{st.session_state.patient_id}</span><span class="session-tag">{source}</span><p class="small-muted" style="margin:10px 0 0">{st.session_state.session_note}</p></div>', unsafe_allow_html=True)

    # Full-width current-window trace: keep the original compact height, widen horizontally.
    if st.session_state.last_signal is not None:
        eeg = np.asarray(st.session_state.last_signal, dtype=float)
        ex = np.arange(1, len(eeg) + 1)
        mini = go.Figure(go.Scatter(x=ex, y=eeg, mode="lines", line=dict(color="#4ee1b5", width=1.8), fill="tozeroy", fillcolor="rgba(78,225,181,.045)"))
        mini.update_layout(template="plotly_dark", height=105, margin=dict(l=5,r=5,t=22,b=3), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#05131b", title=dict(text="LIVE EEG TRACE · CURRENT WINDOW", x=0.01, xanchor="left", font=dict(size=10,color="#7891a1")), xaxis=dict(showgrid=False, showticklabels=False, zeroline=False), yaxis=dict(showgrid=False, showticklabels=False, zeroline=False), showlegend=False)
        st.markdown('<div class="live-current-trace" style="width:100%;">', unsafe_allow_html=True)
        st.plotly_chart(mini, use_container_width=True, config={"displaylogo": False})
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Live EEG trace will appear when a signal window is available.")

        if st.session_state.alerts:
            latest = st.session_state.alerts[0]
            st.markdown(f'<div class="card" style="margin-top:10px;border-color:rgba(255,93,115,.35)"><div class="label">Latest alert event</div><div class="metric" style="color:#ff7a8b">{latest["probability"]:.1%}</div><div class="sub">{latest["time"]} · {latest["source"]}</div></div>', unsafe_allow_html=True)

    st.markdown("### Alert history & report center")
    left2, right2 = st.columns([1.55, 1])
    with left2:
        if st.session_state.alerts:
            adf = pd.DataFrame(st.session_state.alerts)
            st.dataframe(adf, use_container_width=True, hide_index=True, column_config={"probability": st.column_config.NumberColumn("Probability", format="%.1%"), "threshold": st.column_config.NumberColumn("Threshold", format="%.0%")})
        else:
            st.info("No alert events recorded in this session.")
    with right2:
        st.markdown('<div class="card"><div class="section-title">Patient session</div><div class="small-muted">Case</div><b>{}</b><br><div class="small-muted" style="margin-top:8px">Session</div><b>{}</b><br><div class="small-muted" style="margin-top:8px">Model</div><b>{}</b><br><div class="small-muted" style="margin-top:8px">Learned threshold</div><b>{:.1%}</b></div>'.format(st.session_state.patient_id, st.session_state.patient_name, trained_model_name, threshold), unsafe_allow_html=True)
        report = {"report_generated_utc": datetime.now(timezone.utc).isoformat(), "patient_case_id": st.session_state.patient_id, "session_label": st.session_state.patient_name, "note": st.session_state.session_note, "model": model.__class__.__name__, "threshold": threshold, "threshold_method": meta.get("threshold_method", "validation F1 maximization"), "latest_probability": prob, "latest_prediction": pred, "alerts": st.session_state.alerts}
        report_json = json.dumps(report, indent=2)
        report_df = pd.DataFrame(st.session_state.alerts)
        csv_bytes = report_df.to_csv(index=False).encode("utf-8") if not report_df.empty else b"time,patient,probability,threshold,source\n"
        st.download_button("⬇ Download session report (JSON)", report_json, file_name=f"seizure_monitor_{st.session_state.patient_id}.json", mime="application/json", use_container_width=True)
        st.download_button("⬇ Download alert history (CSV)", csv_bytes, file_name=f"alert_history_{st.session_state.patient_id}.csv", mime="text/csv", use_container_width=True)


live_command_center()

# Analytics remains outside the live fragment so it does not redraw every 700 ms.
with st.expander("📊 Model analytics", expanded=False):
    results_df = pd.DataFrame(meta.get("results_df", [])) if isinstance(meta, dict) and meta.get("results_df") is not None else pd.DataFrame()
    validation_df = pd.DataFrame(meta.get("validation_results_df", [])) if isinstance(meta, dict) and meta.get("validation_results_df") is not None else pd.DataFrame()
    roc_auc = float(meta.get("roc_auc", np.nan)) if isinstance(meta, dict) and meta.get("roc_auc") is not None else np.nan
    cm = np.asarray(meta.get("confusion_matrix")) if isinstance(meta, dict) and meta.get("confusion_matrix") is not None else None
    total = 11500; seizure_n = 2300; non_n = 9200

    st.markdown(f'''<div class="analytics-hero"><div><h2>Model performance cockpit</h2><p>Evaluation, decision threshold, class balance and diagnostic views in one place.</p></div><div class="analytics-badge">{trained_model_name} · threshold {threshold:.1%}</div></div>''', unsafe_allow_html=True)
    st.markdown(f'''<div class="analytics-grid">
      <div class="analytics-stat"><div class="n">{total:,}</div><div class="l">Dataset windows</div><div class="s">178 EEG features per window</div></div>
      <div class="analytics-stat"><div class="n">{seizure_n:,}</div><div class="l">Seizure windows</div><div class="s">20% of dataset</div></div>
      <div class="analytics-stat"><div class="n">{non_n:,}</div><div class="l">Non-seizure windows</div><div class="s">80% of dataset</div></div>
      <div class="analytics-stat"><div class="n">{threshold:.0%}</div><div class="l">Alert threshold</div><div class="s">Validation F1 optimized</div></div>
    </div>''', unsafe_allow_html=True)

    at1, at2, at3 = st.tabs(["🏆 Model comparison", "🧪 Diagnostics", "🧬 Feature signals"])
    with at1:
        if not results_df.empty:
            display_df = results_df.copy()
            numeric_cols = [c for c in display_df.columns if str(c).lower() in {"accuracy","precision","recall","f1","roc_auc","roc-auc","auc"}]
            st.dataframe(display_df, use_container_width=True, hide_index=True, column_config={c: st.column_config.NumberColumn(str(c).replace("_"," ").title(), format="%.3f") for c in numeric_cols})
            metric_options = [c for c in display_df.columns if str(c).lower() in {"accuracy","precision","recall","f1","roc_auc","roc-auc","auc"}]
            if metric_options:
                default_idx = next((i for i,c in enumerate(metric_options) if str(c).lower()=="f1"),0)
                metric_col = st.selectbox("Compare metric", metric_options, index=default_idx)
                model_col = next((c for c in display_df.columns if str(c).lower() in {"model","name","model_name"}), None)
                if model_col is not None and metric_col in display_df.columns:
                    plot_df=display_df[[model_col,metric_col]].dropna()
                    fig=go.Figure(go.Bar(x=plot_df[model_col].astype(str),y=plot_df[metric_col].astype(float),marker_color="#35e0c2",text=[f"{v:.3f}" for v in plot_df[metric_col]],textposition="outside"))
                    fig.update_layout(template="plotly_dark",height=340,yaxis=dict(title=str(metric_col).replace("_"," ").title(),range=[0,1.05],gridcolor="#17383f"),xaxis=dict(title="Model"),margin=dict(l=10,r=10,t=35,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#071b22",showlegend=False)
                    st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
        else:
            st.info("Model comparison metadata is not packaged yet. Run the deployment export/retraining step to populate it.")
        st.markdown('<div class="analytics-note">Accuracy alone can hide class-imbalance effects. Use recall, precision, F1, ROC-AUC, the confusion matrix, and the validation-selected threshold together when reviewing this research model.</div>', unsafe_allow_html=True)

    with at2:
        d1,d2=st.columns(2)
        with d1:
            st.markdown('<div class="section-title">Confusion matrix <span>held-out evaluation</span></div>',unsafe_allow_html=True)
            if cm is not None and cm.shape==(2,2):
                fig=go.Figure(go.Heatmap(z=cm,x=["Predicted · Non-Seizure","Predicted · Seizure"],y=["Actual · Non-Seizure","Actual · Seizure"],colorscale=[[0,"#102d35"],[.5,"#167a78"],[1,"#35e0c2"]],text=cm,texttemplate="%{text}",textfont=dict(size=18),hovertemplate="%{y}<br>%{x}<br>Count: %{z}<extra></extra>"))
                fig.update_layout(template="plotly_dark",height=350,margin=dict(l=10,r=10,t=20,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#071b22")
                st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
            else: st.info("Confusion-matrix metadata is not available.")
        with d2:
            st.markdown('<div class="section-title">ROC curve <span>discrimination view</span></div>',unsafe_allow_html=True)
            fpr=meta.get("fpr") if isinstance(meta,dict) else None; tpr=meta.get("tpr") if isinstance(meta,dict) else None
            if fpr is not None and tpr is not None:
                fig=go.Figure(); fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",line=dict(color="#56d9ff",width=3),name=f"AUC {roc_auc:.3f}")); fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",line=dict(color="#6b858b",dash="dash"),showlegend=False))
                fig.update_layout(template="plotly_dark",height=350,xaxis_title="False positive rate",yaxis_title="True positive rate",margin=dict(l=10,r=10,t=20,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#071b22",legend=dict(orientation="h"))
                st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
            else: st.info("ROC metadata is not available.")
        probs=meta.get("best_proba") if isinstance(meta,dict) else None
        if probs is not None:
            st.markdown('<div class="section-title">Prediction probability distribution <span>held-out windows</span></div>',unsafe_allow_html=True)
            fig=go.Figure(go.Histogram(x=np.asarray(probs),nbinsx=35,marker_color="#9b8cff",opacity=.85)); fig.update_layout(template="plotly_dark",height=300,xaxis_title="Seizure probability",yaxis_title="Window count",margin=dict(l=10,r=10,t=20,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#071b22")
            st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
        if not validation_df.empty:
            with st.expander("Validation model-selection metrics"):
                st.dataframe(validation_df,use_container_width=True,hide_index=True)

    with at3:
        imp=meta.get("feature_importances") if isinstance(meta,dict) else None
        if imp is not None:
            imp=np.asarray(imp); idx=np.argsort(imp)[::-1][:20]
            fig=go.Figure(go.Bar(x=imp[idx][::-1],y=[f"X{i+1}" for i in idx][::-1],orientation="h",marker_color="#ffca6b",text=[f"{v:.4f}" for v in imp[idx][::-1]],textposition="outside"))
            fig.update_layout(template="plotly_dark",height=560,xaxis_title="Importance",yaxis_title="EEG feature",margin=dict(l=10,r=70,t=20,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#071b22")
            st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
        else: st.info("Feature importance is available when the selected model exposes feature_importances_.")
        st.markdown('<div class="analytics-note">Feature importance is model-specific. It describes contribution within the trained estimator and should not be interpreted as a clinical biomarker by itself.</div>',unsafe_allow_html=True)

    if source in ("Real seizure replay", "Real non-seizure replay"):
        st.caption("Replay source: real 178-sample EEG windows from the same dataset family used for training. Replay label is shown for demonstration only; model prediction remains independent.")

st.caption("Realtime Seizure Monitoring · Research/demo interface · Model output is not a medical diagnosis. Use appropriate clinical oversight for real-world deployment.")

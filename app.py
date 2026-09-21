from pathlib import Path
from datetime import datetime, timezone
import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Hide Streamlit main menu, header, and footer
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

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

st.markdown(
    """
<style>
:root{--bg:#041018;--panel:#081923;--panel2:#0b202c;--line:#193746;--text:#eff8fb;--muted:#7891a1;--cyan:#5de7e0;--green:#56df9b;--red:#ff5d73;--amber:#ffc45c;--blue:#70a9ff}
.stApp{background:radial-gradient(circle at 78% -10%,rgba(67,173,255,.13),transparent 31%),radial-gradient(circle at 8% 0%,rgba(54,231,214,.09),transparent 27%),var(--bg);color:var(--text)}
.block-container{max-width:1540px;padding:1.1rem 1.7rem 3rem}
section[data-testid="stSidebar"]{background:#05131d;border-right:1px solid #16303e}
.hero{display:flex;justify-content:space-between;align-items:center;gap:20px;padding:23px 27px;border:1px solid #204150;border-radius:23px;background:linear-gradient(135deg,#0a202e,#07131c);box-shadow:0 20px 60px rgba(0,0,0,.22);margin-bottom:30px}
.hero h1{margin:0;font-size:2.5rem;letter-spacing:-.05em}.hero p{margin:.4rem 0 0;color:#8ea7b7}.kicker{font-size:.7rem;letter-spacing:.15em;font-weight:850;color:var(--cyan)}
.live-pill{display:flex;align-items:center;gap:9px;padding:10px 14px;border-radius:999px;border:1px solid #254b55;background:#09242a;font-weight:850;font-size:.8rem;white-space:nowrap}.live-dot{width:9px;height:9px;border-radius:50%;background:var(--green);box-shadow:0 0 0 0 rgba(86,223,155,.55);animation:pulse 1.45s infinite}
@keyframes pulse{70%{box-shadow:0 0 0 9px rgba(86,223,155,0)}100%{box-shadow:0 0 0 0 rgba(86,223,155,0)}}
.card{background:linear-gradient(180deg,rgba(12,31,43,.96),rgba(7,20,29,.96));border:1px solid var(--line);border-radius:17px;padding:16px 17px;margin-bottom:25px !important}.metric{font-size:1.7rem;font-weight:850;line-height:1.1}.label{font-size:.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em}.sub{font-size:.76rem;color:#9bb0bd;margin-top:6px}.small-muted{color:var(--muted);font-size:.74rem}
.alert-banner,.safe-banner{border-radius:17px;padding:14px 18px;margin:12px 0;border:1px solid}.alert-banner{border-color:rgba(255,93,115,.55);background:rgba(255,93,115,.10);animation:alertFlash 1s ease-in-out infinite alternate}.safe-banner{border-color:rgba(86,223,155,.3);background:rgba(86,223,155,.065)}
@keyframes alertFlash{from{box-shadow:0 0 0 rgba(255,93,115,0)}to{box-shadow:0 0 32px rgba(255,93,115,.18)}}
.section-title{font-size:1rem;font-weight:850;margin:4px 0 12px}.section-title span{color:var(--muted);font-size:.78rem;font-weight:500;margin-left:7px}
.session-tag{display:inline-block;padding:5px 9px;border-radius:8px;background:#102a38;color:#a8c0cf;font-size:.7rem;margin-right:5px;border:1px solid #1b4050}
.top-telemetry{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:20px;margin:24px 0}.telemetry{background:rgba(7,23,32,.82);border:1px solid #173746;border-radius:14px;padding:16px 18px;position:relative;overflow:hidden}.telemetry:after{content:"";position:absolute;top:0;bottom:0;width:80px;background:linear-gradient(90deg,transparent,rgba(93,231,224,.09),transparent);animation:sweep 3s linear infinite}.telemetry .k{font-size:.61rem;color:#6e8999;text-transform:uppercase;letter-spacing:.11em}.telemetry .v{font-size:1.02rem;font-weight:850;margin-top:2px}.telemetry .s{font-size:.67rem;color:#91a9b8;margin-top:2px}.live-text{color:var(--green)}.standby-text{color:#9aabb5}
@keyframes sweep{from{left:-100px}to{left:100%}}
.monitor-shell{border:1px solid #183b4a;border-radius:20px;background:linear-gradient(145deg,#06131d,#081c27);padding:13px;box-shadow:inset 0 1px 0 rgba(255,255,255,.025),0 20px 60px rgba(0,0,0,.17);margin-bottom:30px !important}
.monitor-head{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:10px !important}.monitor-head .title{font-size:1.1rem;font-weight:850}.monitor-head .meta{font-size:.78rem;color:#718b9c;margin-top:2px}
.ecg-wrap{margin-top:10px !important;margin-left: -13px !important;margin-right: -13px !important;width: calc(100% + 26px) !important;background:#05131b;border:1px solid #173746;border-radius:13px;overflow:hidden;position:relative}.ecg-label{position:absolute;left:10px;top:8px;font-size:.6rem;letter-spacing:.12em;color:#6f8999;z-index:2}.ecg-svg{display:block;width:100% !important;height:88px}.ecg-path{fill:none;stroke:var(--green);stroke-width:2.3;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:900;stroke-dashoffset:900;animation:draw 2.15s linear infinite}.scan-beam{animation:beam 1.8s linear infinite}@keyframes draw{to{stroke-dashoffset:0}}@keyframes beam{from{transform:translateX(-8px)}to{transform:translateX(1000px)}}
.pulse-line{height:3px;border-radius:4px;background:linear-gradient(90deg,transparent,var(--cyan),transparent);background-size:220% 100%;animation:scan 1.25s linear infinite;opacity:.8}@keyframes scan{to{background-position:-220% 0}}
.alert-dot,.safe-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}.alert-dot{background:var(--red);box-shadow:0 0 0 0 rgba(255,93,115,.5);animation:pulse 1s infinite}.safe-dot{background:var(--green);box-shadow:0 0 0 0 rgba(86,223,155,.5);animation:pulse 1.7s infinite}
.stButton>button{border-radius:11px;border:1px solid #274a5a;background:#0c2533;color:#edf8fa;font-weight:800;min-height:43px}.stButton>button:hover{border-color:var(--cyan);background:#103342}
[data-testid="stMetric"]{background:transparent}footer{visibility:hidden}
@media(max-width:900px){.hero h1{font-size:1.8rem}.top-telemetry{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not MODEL_PATH.exists():
        return None, None, {}
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    meta = joblib.load(META_PATH) if META_PATH.exists() else {}
    return model, scaler, meta


model, scaler, meta = load_artifacts()


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
        "threshold": 0.50,
        "source": "Demo streaming",
        "source_detail": "Synthetic demonstration stream",
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
    <p>Hospital-style EEG command center for live screening, event surveillance and session reporting.</p>
  </div>
  <div class="live-pill"><span class="live-dot"></span> MONITOR READY</div>
</div>
""",
    unsafe_allow_html=True,
)

if model is None or scaler is None:
    st.error("Model artifacts are missing. Add outputs/best_model.pkl and outputs/scaler.pkl to the deployment repository.")
    st.stop()


# Sidebar
st.sidebar.markdown("## 🏥 Control Room")
st.sidebar.caption("Realtime Seizure Monitoring")
patient_id = st.sidebar.text_input("Patient / Case ID", value=st.session_state.patient_id)
patient_name = st.sidebar.text_input("Session label", value=st.session_state.patient_name)
session_note = st.sidebar.text_area("Clinical / research note", value=st.session_state.session_note, height=80)
threshold = st.sidebar.slider("Alert threshold", 0.10, 0.95, float(st.session_state.threshold), 0.05)
source = st.sidebar.selectbox("Stream source", ["Demo streaming", "EEG CSV", "Single EEG window"], index=["Demo streaming", "EEG CSV", "Single EEG window"].index(st.session_state.source))
st.session_state.update(patient_id=patient_id, patient_name=patient_name, session_note=session_note, threshold=threshold, source=source)

if st.sidebar.button("🧹 Clear alert history", use_container_width=True):
    st.session_state.alerts = []
    st.session_state.previous_prediction = 0

st.sidebar.markdown("---")
st.sidebar.markdown("**Safety**")
st.sidebar.caption("Research/demo interface. Alerts are model outputs and must not be treated as a diagnosis or emergency medical decision.")


def get_csv_signal():
    uploaded = st.file_uploader("Upload a CSV containing one or more 178-point EEG windows", type=["csv"])
    if uploaded is None:
        return None, None
    df = pd.read_csv(uploaded)
    feature_cols = [c for c in df.columns if str(c).lower().startswith("x")]
    if len(feature_cols) != 178:
        feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and str(c).lower() not in {"y", "target", "label"}]
    if len(feature_cols) != 178:
        st.error(f"Expected 178 EEG columns (X1–X178); detected {len(feature_cols)} usable columns.")
        return None, None
    st.session_state.csv_df = df
    st.session_state.csv_features = feature_cols
    max_row = max(len(df) - 1, 0)
    row = st.slider("Preview / starting row", 0, max_row, min(st.session_state.csv_row, max_row))
    st.session_state.csv_row = row
    return df[feature_cols].iloc[row].astype(float).values, f"CSV row {row + 1}/{len(df)}"


def parse_single():
    text = st.text_area("Paste exactly 178 EEG values", height=120, placeholder="0.14, -0.32, 0.51, ...")
    if not text.strip():
        return None, None
    try:
        signal = np.fromstring(text.replace(",", " ").replace("\n", " "), sep=" ")
        if len(signal) != 178:
            st.warning(f"Received {len(signal)} values. Exactly 178 are required.")
            return None, None
        return signal, "Manual EEG window"
    except Exception:
        st.error("Could not parse the EEG values.")
        return None, None


def demo_signal(tick=0):
    rng = np.random.default_rng(42 + tick)
    t = np.linspace(0, 1, 178)
    active = tick % 22 in range(8, 15)
    burst = 1.0 + (0.72 * np.exp(-((t - (0.26 + 0.03 * np.sin(tick / 5))) / 0.075) ** 2) if active else 0.0)
    return burst * (0.72 * np.sin(2 * np.pi * (8.5 + 0.2 * np.sin(tick / 7)) * t) + 0.25 * np.sin(2 * np.pi * 17 * t)) + rng.normal(0, .22, 178)


if source == "EEG CSV":
    signal, source_detail = get_csv_signal()
elif source == "Single EEG window":
    signal, source_detail = parse_single()
else:
    signal, source_detail = demo_signal(st.session_state.tick), "Synthetic demonstration stream"


def predict(signal):
    x = np.asarray(signal, dtype=float).reshape(1, -1)
    if x.shape[1] != 178:
        raise ValueError(f"Model expects 178 features, received {x.shape[1]}.")
    scaled = scaler.transform(x)
    proba = float(model.predict_proba(scaled)[0, 1])
    pred = int(proba >= st.session_state.threshold)
    return pred, proba


# Controls
c1, c2, c3, c4 = st.columns([1.05, 1.05, 1.05, 2.85])
with c1:
    if st.button("▶ Start monitoring", use_container_width=True, disabled=st.session_state.monitoring):
        st.session_state.monitoring = True
        st.session_state.started_at = datetime.now(timezone.utc)
        st.session_state.tick = 0
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
        if source == "Demo streaming":
            live_signal = demo_signal(tick)
            detail = "Synthetic demonstration stream"
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
    pulse = 72 + ((st.session_state.tick * 3) % 9)
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
      <div class="telemetry"><div class="k">Alert threshold</div><div class="v">{threshold:.0%}</div><div class="s">Current session setting</div></div>
    </div>''', unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f'<div class="card"><div class="label">Patient / case</div><div class="metric">{st.session_state.patient_id}</div><div class="sub">{st.session_state.patient_name}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="card"><div class="label">Seizure probability</div><div class="metric">{prob:.1%}</div><div class="sub">Threshold {threshold:.0%}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="card"><div class="label">Monitor pulse</div><div class="metric">{pulse} <span style="font-size:.8rem">bpm</span></div><div class="sub">Visual monitor effect · not ECG-derived</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="card"><div class="label">Alert events</div><div class="metric">{len(st.session_state.alerts):02d}</div><div class="sub">This monitoring session</div></div>', unsafe_allow_html=True)
    with m5:
        st.markdown(f'<div class="card"><div class="label">Session time</div><div class="metric">{elapsed//60:02d}:{elapsed%60:02d}</div><div class="sub">{model.__class__.__name__}</div></div>', unsafe_allow_html=True)

    left, right = st.columns([2.15, 1])
    with left:
        st.markdown('<div class="monitor-shell"><div class="monitor-head"><div><div class="title">Live EEG waveform</div><div class="meta">Streaming neural signal · 178 samples · model input window</div></div><span class="session-tag">LIVE EEG</span></div>', unsafe_allow_html=True)
        if st.session_state.last_signal is not None:
            y = np.asarray(st.session_state.last_signal)
            x = np.arange(1, len(y) + 1)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="#5de7e0", width=2.3), fill="tozeroy", fillcolor="rgba(93,231,224,.035)", hovertemplate="Sample %{x}<br>Amplitude %{y:.4f}<extra></extra>"))
            fig.update_layout(template="plotly_dark", height=400, margin=dict(l=5,r=5,t=8,b=5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#06131b", xaxis=dict(title="Sample", gridcolor="#173040", zeroline=False), yaxis=dict(title="EEG amplitude", gridcolor="#173040", zeroline=False), showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": False})
        else:
            st.info("Load an EEG source and start monitoring to populate the live waveform.")
        st.markdown('''<div class="ecg-wrap"><div class="ecg-label">VISUAL PULSE TRACE · MONITOR TELEMETRY</div><svg class="ecg-svg" viewBox="0 0 760 88" preserveAspectRatio="none"><path class="ecg-path" d="M0 52 L75 52 L92 50 L106 54 L122 52 L142 52 L154 17 L165 75 L176 43 L194 52 L270 52 L287 50 L301 54 L317 52 L337 52 L349 17 L360 75 L371 43 L389 52 L465 52 L482 50 L496 54 L512 52 L532 52 L544 17 L555 75 L566 43 L584 52 L660 52 L677 50 L691 54 L707 52 L727 52 L739 17 L750 75 L761 43 L780 52"/><rect class="scan-beam" x="0" y="0" width="3" height="88" fill="rgba(93,231,224,.58)"/></svg></div><div class="pulse-line"></div></div>''', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-title">Signal status <span>model output</span></div>', unsafe_allow_html=True)
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=prob*100, number={"suffix":"%", "font":{"size":38}}, title={"text":"Seizure probability"}, gauge={"axis":{"range":[0,100]}, "bar":{"color":"#ff5d73" if pred else "#56df9b", "thickness":.28}, "steps":[{"range":[0,threshold*100],"color":"rgba(86,223,155,.10)"},{"range":[threshold*100,100],"color":"rgba(255,93,115,.10)"}], "threshold":{"line":{"color":"#ffc45c","width":4},"value":threshold*100}}))
        gauge.update_layout(template="plotly_dark", height=270, margin=dict(l=10,r=10,t=25,b=5), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(gauge, use_container_width=True, config={"displaylogo": False})
        st.markdown(f'<div class="card"><span class="session-tag">{st.session_state.patient_id}</span><span class="session-tag">{source}</span><p class="small-muted" style="margin:10px 0 0">{st.session_state.session_note}</p></div>', unsafe_allow_html=True)
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
        st.markdown('<div class="card"><div class="section-title">Patient session</div><div class="small-muted">Case</div><b>{}</b><br><div class="small-muted" style="margin-top:8px">Session</div><b>{}</b><br><div class="small-muted" style="margin-top:8px">Model</div><b>{}</b></div>'.format(st.session_state.patient_id, st.session_state.patient_name, model.__class__.__name__), unsafe_allow_html=True)
        report = {"report_generated_utc": datetime.now(timezone.utc).isoformat(), "patient_case_id": st.session_state.patient_id, "session_label": st.session_state.patient_name, "note": st.session_state.session_note, "model": model.__class__.__name__, "threshold": threshold, "latest_probability": prob, "latest_prediction": pred, "alerts": st.session_state.alerts}
        report_json = json.dumps(report, indent=2)
        report_df = pd.DataFrame(st.session_state.alerts)
        csv_bytes = report_df.to_csv(index=False).encode("utf-8") if not report_df.empty else b"time,patient,probability,threshold,source\n"
        st.download_button("⬇ Download session report (JSON)", report_json, file_name=f"seizure_monitor_{st.session_state.patient_id}.json", mime="application/json", use_container_width=True)
        st.download_button("⬇ Download alert history (CSV)", csv_bytes, file_name=f"alert_history_{st.session_state.patient_id}.csv", mime="text/csv", use_container_width=True)


live_command_center()

# Analytics remains outside the live fragment so it does not redraw every 700 ms.
with st.expander("📊 Model analytics", expanded=False):
    if isinstance(meta, dict) and meta.get("results_df") is not None:
        try:
            st.dataframe(pd.DataFrame(meta["results_df"]), use_container_width=True)
        except Exception:
            pass
    tabs = st.tabs(["Confusion Matrix", "ROC", "Probability Distribution", "Feature Importance"])
    with tabs[0]:
        cm = meta.get("confusion_matrix") if isinstance(meta, dict) else None
        if cm is not None:
            cm = np.asarray(cm)
            fig = go.Figure(go.Heatmap(z=cm, x=["Non-Seizure","Seizure"], y=["Non-Seizure","Seizure"], colorscale="Teal", text=cm, texttemplate="%{text}"))
            fig.update_layout(template="plotly_dark", height=360, margin=dict(l=10,r=10,t=25,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run the deployment export cell in the notebook to package evaluation metadata.")
    with tabs[1]:
        fpr = meta.get("fpr") if isinstance(meta, dict) else None
        tpr = meta.get("tpr") if isinstance(meta, dict) else None
        if fpr is not None and tpr is not None:
            fig = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color="#70a9ff", width=3), name=f"AUC {meta.get('roc_auc',0):.3f}"))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(color="#526473", dash="dash"), showlegend=False))
            fig.update_layout(template="plotly_dark", height=360, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", margin=dict(l=10,r=10,t=25,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ROC metadata is not packaged yet.")
    with tabs[2]:
        probs = meta.get("best_proba") if isinstance(meta, dict) else None
        if probs is not None:
            fig = go.Figure(go.Histogram(x=np.asarray(probs), nbinsx=35, marker_color="#5de7e0"))
            fig.update_layout(template="plotly_dark", height=360, xaxis_title="Seizure probability", yaxis_title="Count", margin=dict(l=10,r=10,t=25,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Probability metadata is not packaged yet.")
    with tabs[3]:
        imp = meta.get("feature_importances") if isinstance(meta, dict) else None
        if imp is not None:
            imp = np.asarray(imp)
            idx = np.argsort(imp)[::-1][:20]
            fig = go.Figure(go.Bar(x=imp[idx][::-1], y=[f"X{i+1}" for i in idx][::-1], orientation="h", marker_color="#ffc45c"))
            fig.update_layout(template="plotly_dark", height=500, xaxis_title="Importance", yaxis_title="EEG feature", margin=dict(l=10,r=10,t=25,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance is available when the selected model exposes feature_importances_.")

st.caption("Realtime Seizure Monitoring · Research/demo interface · Model output is not a medical diagnosis. Use appropriate clinical oversight for real-world deployment.")

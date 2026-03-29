import os
import re
import random
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
    AUTO_REFRESH_AVAILABLE = True
except Exception:
    AUTO_REFRESH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AegisAI – Data Center Copilot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;800&display=swap');
html,body,[class*="css"]{font-family:'Exo 2',sans-serif !important;}
header[data-testid="stHeader"]{background:transparent !important;height:0rem !important;}
[data-testid="stToolbar"]{right:1rem !important;top:0.5rem !important;}
.block-container{padding-top:0.35rem !important;padding-bottom:2rem !important;max-width:1500px;}
.stApp{
  background:
    radial-gradient(ellipse at 10% 0%,rgba(0,255,255,0.12) 0%,transparent 45%),
    radial-gradient(ellipse at 90% 0%,rgba(120,40,255,0.14) 0%,transparent 45%),
    radial-gradient(ellipse at 50% 100%,rgba(0,180,255,0.10) 0%,transparent 45%),
    linear-gradient(160deg,#071424 0%,#0b1d34 45%,#0a1730 100%);
  color:#eaf4ff;
}
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#08172a 0%,#0b1d34 100%) !important;
  border-right:1px solid rgba(0,255,255,0.12) !important;
}
[data-testid="stSidebar"] .block-container{padding:1rem !important;}
.glass{
  background:rgba(255,255,255,0.05);
  border:1px solid rgba(255,255,255,0.10);
  backdrop-filter:blur(20px);
  border-radius:20px;
  padding:18px 20px;
  margin-bottom:14px;
  position:relative;
  overflow:hidden;
}
.glass::before{
  content:'';
  position:absolute;
  top:0;left:0;right:0;
  height:1px;
  background:linear-gradient(90deg,transparent,rgba(0,255,255,0.35),transparent);
}
.glass-cyan{border-color:rgba(0,255,255,0.18);box-shadow:0 0 30px rgba(0,255,255,0.05) inset;}
.kpi{
  background:rgba(255,255,255,0.055);
  border:1px solid rgba(255,255,255,0.10);
  border-radius:18px;
  padding:16px 18px;
  min-height:112px;
  position:relative;
  overflow:hidden;
}
.kpi::after{
  content:'';
  position:absolute;
  bottom:0;left:0;right:0;
  height:2px;
  border-radius:0 0 18px 18px;
}
.kpi-cyan::after{background:linear-gradient(90deg,transparent,#00ffff,transparent);}
.kpi-red::after{background:linear-gradient(90deg,transparent,#ff3c3c,transparent);}
.kpi-amber::after{background:linear-gradient(90deg,transparent,#ffb400,transparent);}
.kpi-green::after{background:linear-gradient(90deg,transparent,#00dc82,transparent);}
.kpi-label{
  font-family:'Share Tech Mono',monospace !important;
  font-size:10px;
  letter-spacing:0.2em;
  color:#7a94b5;
  text-transform:uppercase;
  margin-bottom:8px;
}
.kpi-value{
  font-family:'Rajdhani',sans-serif !important;
  font-size:38px;
  font-weight:700;
  color:white;
  line-height:1;
}
.kpi-unit{font-size:16px;color:#8ca8c8;font-weight:400;}
.kpi-trend{font-family:'Share Tech Mono',monospace !important;font-size:11px;margin-top:6px;}
.trend-up{color:#ff6666;}
.trend-down{color:#00e590;}
.trend-flat{color:#8ca8c8;}
.sec-title{
  font-family:'Rajdhani',sans-serif !important;
  font-size:18px;
  font-weight:700;
  color:white;
  letter-spacing:0.04em;
  margin-bottom:2px;
}
.sec-sub{
  font-family:'Share Tech Mono',monospace !important;
  font-size:10px;
  color:#8aa3c3;
  letter-spacing:0.14em;
  margin-bottom:12px;
}
.pill{
  display:inline-block;
  padding:4px 12px;
  border-radius:999px;
  font-family:'Share Tech Mono',monospace !important;
  font-size:10px;
  letter-spacing:0.1em;
  margin-right:6px;
  border:1px solid;
}
.pill-ok{background:rgba(0,220,130,0.12);color:#00dc82;border-color:rgba(0,220,130,0.30);}
.pill-crit{background:rgba(255,60,60,0.12);color:#ff5555;border-color:rgba(255,60,60,0.30);}
.pill-info{background:rgba(0,200,255,0.12);color:#00c8ff;border-color:rgba(0,200,255,0.30);}
.pill-fixed{background:rgba(120,80,255,0.12);color:#a050ff;border-color:rgba(120,80,255,0.30);}
.pill-warn{background:rgba(255,180,0,0.12);color:#ffb400;border-color:rgba(255,180,0,0.30);}
.ai-insight{
  background:linear-gradient(135deg,rgba(0,200,255,0.08),rgba(120,40,255,0.08));
  border:1px solid rgba(0,200,255,0.18);
  border-radius:16px;
  padding:14px 16px;
  font-size:13px;
  line-height:1.7;
  color:#d1e6ff;
  margin-bottom:12px;
}
.rec{
  background:rgba(255,255,255,0.04);
  border:1px solid rgba(255,255,255,0.08);
  border-radius:14px;
  padding:11px 14px;
  margin-bottom:8px;
}
.rec-title{font-weight:700;color:white;font-size:13px;margin-bottom:3px;}
.rec-desc{color:#bfd5ee;font-size:11.5px;line-height:1.5;}
.xai-bar-wrap{margin-bottom:10px;}
.xai-label{
  font-family:'Share Tech Mono',monospace;
  font-size:10px;
  color:#8aa3c3;
  letter-spacing:0.1em;
  display:flex;
  justify-content:space-between;
  margin-bottom:4px;
}
.xai-bar-bg{background:rgba(255,255,255,0.06);border-radius:6px;height:10px;overflow:hidden;}
.xai-bar-fill{height:100%;border-radius:6px;}
.chat-wrap{
  display:flex;
  flex-direction:column;
  gap:10px;
  max-height:320px;
  overflow-y:auto;
  padding-right:4px;
  margin-bottom:12px;
}
.chat-bubble{padding:10px 14px;border-radius:14px;font-size:12.5px;line-height:1.6;max-width:92%;}
.chat-user{
  background:rgba(0,200,255,0.12);
  border:1px solid rgba(0,200,255,0.22);
  color:#d9f0ff;
  align-self:flex-end;
  border-radius:14px 14px 4px 14px;
}
.chat-bot{
  background:rgba(120,80,255,0.10);
  border:1px solid rgba(120,80,255,0.22);
  color:#e2d9ff;
  align-self:flex-start;
  border-radius:14px 14px 14px 4px;
}
.chat-ts{font-family:'Share Tech Mono',monospace;font-size:9px;color:#8aa3c3;margin-top:3px;}
div[data-baseweb="select"]>div{
  background:rgba(255,255,255,0.05) !important;
  border-color:rgba(0,255,255,0.14) !important;
  border-radius:12px !important;
  color:white !important;
}
.stButton>button{
  background:rgba(0,200,255,0.10) !important;
  border:1px solid rgba(0,200,255,0.24) !important;
  border-radius:12px !important;
  color:#00d5ff !important;
  font-family:'Rajdhani',sans-serif !important;
  font-weight:600 !important;
  font-size:13px !important;
  letter-spacing:0.06em !important;
}
.stButton>button:hover{
  background:rgba(0,200,255,0.18) !important;
  border-color:rgba(0,200,255,0.42) !important;
}
[data-testid="metric-container"]{display:none;}
div[data-testid="stExpander"]{
  background:rgba(255,255,255,0.03) !important;
  border:1px solid rgba(255,255,255,0.08) !important;
  border-radius:14px !important;
}
.stTextInput label{display:none !important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "scenario": "Latest / Live",
    "selected_zone": "All Zones",
    "time_range": "Last 30 min",
    "live_pointer": 0,
    "what_if_ran": False,
    "what_if_report": None,
    "chat_history": [],
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    margin=dict(l=10, r=10, t=10, b=10),
)

ZONE_MAP = {
    "R-01": "Zone A", "R-02": "Zone A", "R-03": "Zone B", "R-04": "Zone B",
    "R-05": "Zone C", "R-06": "Zone C", "R-07": "Zone D", "R-08": "Zone D",
}
RACK_POS = {
    "R-01": (1, 1), "R-02": (2, 1), "R-03": (3, 1), "R-04": (4, 1),
    "R-05": (1, 2), "R-06": (2, 2), "R-07": (3, 2), "R-08": (4, 2),
}
TIME_RANGES = {
    "Last 10 min": 10,
    "Last 30 min": 30,
    "Last 1 hr": 60,
    "Last 6 hr": 360,
}
HEATMAP_RACK_COLORS = ["#1f77ff", "#00c853", "#ffd600", "#ff2d2d"]  # blue, green, yellow, red


# ─────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str).str.strip().str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    return df

def find_csv(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

@st.cache_data(show_spinner=False)
def load_base_dataframe():
    path = find_csv(["data/processed.csv", "processed.csv", "data/sample_csv/processed.csv"])
    if not path:
        raise FileNotFoundError("Put CSV at data/processed.csv or processed.csv.")

    df = clean_cols(pd.read_csv(path))

    for s, d in [
        ("cpu_load", "cpu"),
        ("temperature", "temp"),
        ("power_draw", "power"),
        ("net", "network"),
    ]:
        if s in df.columns and d not in df.columns:
            df = df.rename(columns={s: d})

    req = ["cpu", "temp", "power", "network"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"CSV missing: {miss}")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if df["timestamp"].notna().sum() > 0:
            df = df.sort_values("timestamp").reset_index(drop=True)
        else:
            df["timestamp"] = pd.date_range(
                end=pd.Timestamp.now().floor("s"),
                periods=len(df),
                freq="s"
            )
    else:
        df["timestamp"] = pd.date_range(
            end=pd.Timestamp.now().floor("s"),
            periods=len(df),
            freq="s"
        )

    for c in req:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=req).reset_index(drop=True)
    if df.empty:
        raise ValueError("No usable rows after cleaning.")
    return df, path

def norm(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    lo, hi = np.nanpercentile(s, 5), np.nanpercentile(s, 95)
    if hi - lo <= 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return ((s - lo) / (hi - lo)).clip(0, 1)

@st.cache_data(show_spinner=False)
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    w = df.copy()
    for c, nc in [("temp", "temp_norm"), ("cpu", "cpu_norm"), ("power", "power_norm"), ("network", "network_norm")]:
        w[nc] = norm(w[c])

    w["anomaly_score"] = (
        0.35 * w["temp_norm"] +
        0.30 * w["cpu_norm"] +
        0.20 * w["power_norm"] +
        0.15 * w["network_norm"]
    ).clip(0, 1)

    win = max(60, min(900, len(w) // 40))
    w["rolling_anomaly"] = w["anomaly_score"].rolling(
        win,
        min_periods=max(10, win // 6)
    ).mean()
    return w

def scenario_windows(df: pd.DataFrame):
    v = df["rolling_anomaly"].fillna(df["anomaly_score"])
    f = int(v.idxmax())
    s = int(v.idxmin())
    r = min(len(df) - 1, f + max(300, len(df) // 120))
    sp = max(1500, min(9000, len(df) // 8))

    def ss(c):
        a = max(0, c - sp // 2)
        b = min(len(df), a + sp)
        a = max(0, b - sp)
        return a, b

    return {
        "Latest / Live": (max(0, len(df) - sp), len(df)),
        "Stable Window": ss(s),
        "Failure Window": ss(f),
        "Recovery Window": ss(r),
    }

def slice_tr(df: pd.DataFrame, label: str) -> pd.DataFrame:
    et = df["timestamp"].max()
    st_ = et - timedelta(minutes=TIME_RANGES[label])
    sub = df[df["timestamp"] >= st_].copy()
    if len(sub) >= 10:
        return sub.reset_index(drop=True)

    fb = {"Last 10 min": 300, "Last 30 min": 900, "Last 1 hr": 1800, "Last 6 hr": 7200}
    return df.tail(min(fb[label], len(df))).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# METRIC SCALING
# ─────────────────────────────────────────────────────────────────────────────
def sc_temp(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-6:
        return s * 0 + 21.0
    return (18.0 + (s - lo) / (hi - lo) * 6.0).clip(18, 24)

def sc_cpu(s: pd.Series) -> pd.Series:
    s = s.astype(float).clip(lower=0)
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-6:
        return s * 0 + 62.0
    return (50.0 + (s - lo) / (hi - lo) * 25.0).clip(50, 75)

def sc_power(s: pd.Series) -> pd.Series:
    s = s.astype(float).clip(lower=0)
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-6:
        return s * 0 + 60.0
    return (35.0 + (s - lo) / (hi - lo) * 55.0).clip(35, 90)

def sc_net(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-6:
        return s * 0 + 50.0
    return ((s - lo) / (hi - lo) * 100.0).clip(0, 100)

def scale_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["temp_d"] = sc_temp(d["temp"])
    d["cpu_d"] = sc_cpu(d["cpu"])
    d["power_d"] = sc_power(d["power"])
    d["net_d"] = sc_net(d["network"])
    return d


# ─────────────────────────────────────────────────────────────────────────────
# XAI
# ─────────────────────────────────────────────────────────────────────────────
def xai(wdf: pd.DataFrame):
    feats = {
        "Temperature": ("temp_norm", 0.35),
        "CPU Load": ("cpu_norm", 0.30),
        "Power Draw": ("power_norm", 0.20),
        "Network I/O": ("network_norm", 0.15),
    }
    tail = wdf.tail(min(len(wdf), 600))
    sc = {}
    for lbl, (col, w) in feats.items():
        c = abs(float(tail[col].corr(tail["anomaly_score"]))) if col in tail.columns else w
        c = 0.0 if np.isnan(c) else c
        sc[lbl] = c * w
    tot = sum(sc.values()) or 1.0
    return {k: round(v / tot * 100, 1) for k, v in sorted(sc.items(), key=lambda x: -x[1])}

def xai_html(att: dict) -> str:
    cm = {
        "Temperature": "#00e5ff",
        "CPU Load": "#a855f7",
        "Power Draw": "#22c55e",
        "Network I/O": "#fb7185",
    }
    h = ""
    for f, p in att.items():
        c = cm.get(f, "#8aa3c3")
        h += (
            f'<div class="xai-bar-wrap">'
            f'<div class="xai-label"><span>{f}</span>'
            f'<span style="color:{c};font-weight:700;">{p}%</span></div>'
            f'<div class="xai-bar-bg"><div class="xai-bar-fill" '
            f'style="width:{p}%;background:{c};opacity:0.85;"></div></div>'
            f'</div>'
        )
    return h


# ─────────────────────────────────────────────────────────────────────────────
# RACK / ZONE
# ─────────────────────────────────────────────────────────────────────────────
def get_rack_box_color(rack_id: str) -> str:
    rng = random.Random(rack_id)
    return rng.choice(HEATMAP_RACK_COLORS)

def build_racks(wdf: pd.DataFrame):
    ids = [f"R-{i:02d}" for i in range(1, 9)]
    tail = wdf.tail(min(len(wdf), 800)).reset_index(drop=True)

    if len(tail) < 8:
        tail = pd.concat([tail] * (8 // max(1, len(tail)) + 1), ignore_index=True).head(8)

    tail["rs"] = tail.index % 8
    rdf = tail.groupby("rs")[["temp", "cpu", "power", "network", "anomaly_score"]].mean().reset_index(drop=True)

    racks = []
    for i, rid in enumerate(ids):
        row = rdf.iloc[i]
        anom = float(row["anomaly_score"])
        status = "critical" if anom >= 0.82 else "warning" if anom >= 0.58 else "healthy"

        t = float(sc_temp(pd.Series([float(row["temp"]) + (i % 4 - 1.5) * 0.30])).iloc[0])
        c = float(sc_cpu(pd.Series([float(row["cpu"]) + (i % 3 - 1.0) * 1.20])).iloc[0])
        c = float(np.clip(c, 50, 75))
        p = float(sc_power(pd.Series([float(row["power"]) + (i % 2 - 0.5) * 1.50])).iloc[0])
        n = float(sc_net(pd.Series([float(row["network"]) + ((i + 1) % 4 - 1.5) * 1.0])).iloc[0])

        racks.append({
            "id": rid,
            "zone": ZONE_MAP[rid],
            "temp": round(t, 1),
            "load": round(c, 1),
            "power": round(p, 1),
            "network": round(n, 1),
            "anomaly": round(anom * 100, 1),
            "status": status,
        })
    return racks

def compute_scores(wdf: pd.DataFrame, scenario: str):
    tn = min(len(wdf), 600)
    aa = float(wdf["anomaly_score"].tail(tn).mean())
    ts = float(norm(wdf["temp"]).tail(tn).mean())
    cs = float(norm(wdf["cpu"]).tail(tn).mean())
    ps = float(norm(wdf["power"]).tail(tn).mean())

    risk = int(np.clip((0.45 * aa + 0.25 * ts + 0.20 * cs + 0.10 * ps) * 100, 5, 99))
    health = int(np.clip(100 - 0.9 * risk, 5, 99))

    lmap = {
        "Failure Window": ("CRITICAL WINDOW", "pill-crit"),
        "Recovery Window": ("RECOVERY WINDOW", "pill-fixed"),
        "Stable Window": ("STABLE WINDOW", "pill-ok"),
    }
    label, pill = lmap.get(scenario, ("LIVE WINDOW", "pill-info"))

    sd = scale_df(wdf)
    last = sd.iloc[-1]

    return {
        "health": health,
        "risk": risk,
        "fp": risk,
        "label": label,
        "pill": pill,
        "lt": round(float(last["temp_d"]), 1),
        "lc": round(float(last["cpu_d"]), 1),
        "lp": round(float(last["power_d"]), 1),
        "ln": round(float(last["net_d"]), 1),
    }

def zone_risk(racks):
    z = {"A": [], "B": [], "C": [], "D": []}
    for r in racks:
        k = r["zone"].replace("Zone ", "")
        z[k].append(0.50 * r["anomaly"] + 0.30 * (r["temp"] - 18) / 6 * 100 + 0.20 * r["load"])
    return {k: int(np.clip(np.mean(v) / 1.7 if v else 0, 0, 100)) for k, v in z.items()}

def delta_html(a, b):
    d = float(a) - float(b)
    if abs(d) < 0.05:
        return '<span class="kpi-trend trend-flat">— STABLE</span>'
    sym = "▲" if d > 0 else "▼"
    return f'<span class="kpi-trend {"trend-up" if d > 0 else "trend-down"}">{sym} {abs(d):.1f}</span>'

def ai_insight(racks, scenario, att):
    h = max(racks, key=lambda x: x["temp"])
    b = max(racks, key=lambda x: x["load"])
    tf, tp = next(iter(att.items()))
    pfx = {"Failure Window": "⚠", "Recovery Window": "✓", "Stable Window": "✦"}.get(scenario, "ℹ")
    return (
        f"{pfx} Hottest rack: {h['id']} ({h['zone']}) at <b>{h['temp']}°C</b>. "
        f"Busiest: {b['id']} at <b>{b['load']:.0f}%</b> CPU. "
        f"XAI primary driver: <b>{tf}</b> at {tp}%."
    )

def recommendations(racks, scenario, att):
    h = max(racks, key=lambda x: x["temp"])
    b = max(racks, key=lambda x: x["load"])
    tf = next(iter(att.keys()))

    if scenario == "Failure Window":
        return [
            ("CRITICAL", "#ff3c3c", "rgba(255,60,60,0.10)", f"Reduce load on {b['id']}",
             f"XAI: {tf} is primary driver. Migrate workload from {b['id']} to lower-load racks."),
            ("CRITICAL", "#ff3c3c", "rgba(255,60,60,0.10)", f"Cool {h['zone']}",
             f"Raise CRAC setpoint near {h['id']} ({h['temp']}°C). Verify airflow path."),
        ]
    if scenario == "Recovery Window":
        return [
            ("RESOLVED", "#00dc82", "rgba(0,220,130,0.08)", "Maintain balanced workload",
             f"XAI: {tf} still elevated. Keep distribution until {h['id']} stabilises."),
        ]
    if scenario == "Stable Window":
        return [
            ("ADVISORY", "#00a0e0", "rgba(0,160,224,0.08)", "Preventive load balancing",
             f"XAI flags {tf} as latent risk. Pre-balance compute before peak windows."),
        ]
    return []

def build_what_if(scores, racks, scenario, att):
    h = max(racks, key=lambda x: x["temp"])
    b = max(racks, key=lambda x: x["load"])
    tf = next(iter(att.keys()))
    br = scores["risk"]

    if scenario == "Failure Window":
        ar = max(18, br - 42)
        ah = min(94, scores["health"] + 28)
        acts = [
            f"Migrate workload from {b['id']} — XAI: {tf} primary driver.",
            f"Increase cooling near {h['zone']} / {h['id']} ({h['temp']}°C ambient).",
            "Throttle non-essential compute during peak anomaly interval.",
            "Re-verify power & fan response post-migration.",
        ]
    elif scenario == "Recovery Window":
        ar = max(12, br - 12)
        ah = min(96, scores["health"] + 10)
        acts = [
            "Hold balanced workload for one more cycle.",
            f"Monitor {h['id']} for residual thermal rebound.",
            "Close incident only after sustained normalisation.",
        ]
    else:
        ar = max(10, br - 8)
        ah = min(98, scores["health"] + 6)
        acts = [
            f"Preventive balancing — XAI flags {tf} as latent risk.",
            "Schedule thermal-path inspection.",
            "Tune anomaly thresholds for early-warning sensitivity.",
        ]

    return {
        "before_risk": br,
        "after_risk": ar,
        "before_health": scores["health"],
        "after_health": ah,
        "hottest_rack": h["id"],
        "hottest_temp": h["temp"],
        "busiest_rack": b["id"],
        "busiest_load": b["load"],
        "top_feat": tf,
        "actions": acts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHATBOT
# ─────────────────────────────────────────────────────────────────────────────
def chatbot(msg, scores, racks, att, scenario):
    m = msg.lower()
    h = max(racks, key=lambda x: x["temp"])
    b = max(racks, key=lambda x: x["load"])
    tf, tp = next(iter(att.items()))

    if any(g in m for g in ["hi", "hello", "hey", "howdy"]):
        return (
            f"Hello! AegisAI Copilot 🛡️ — monitoring {scenario}. "
            f"Health: {scores['health']}/100, Risk: {scores['risk']}%. What would you like to know?"
        )
    if any(k in m for k in ["temp", "thermal", "hot", "heat", "cool", "ambient"]):
        return (
            f"🌡️ Hottest rack: <b>{h['id']}</b> ({h['zone']}) at <b>{h['temp']}°C</b> ambient. "
            f"Temperature contributes {att.get('Temperature', '–')}% to anomaly score. "
            f"{'⚠ Above 23°C — inspect CRAC units.' if h['temp'] > 23 else '✓ Within 18–24°C spec.'}"
        )
    if any(k in m for k in ["cpu", "load", "compute", "utiliz", "process"]):
        return (
            f"⚙️ Busiest rack: <b>{b['id']}</b> at <b>{b['load']:.1f}%</b> CPU. "
            f"CPU Load contributes {att.get('CPU Load', '–')}% to anomaly score. "
            f"{'⚠ High load — consider workload migration.' if b['load'] > 70 else '✓ CPU load nominal.'}"
        )
    if any(k in m for k in ["network", "latency", "ms", "ping", "bandwidth", "net"]):
        mn = max(racks, key=lambda x: x["network"])
        return (
            f"🌐 Highest latency: <b>{mn['id']}</b> at <b>{mn['network']:.1f}ms</b>. "
            f"Network I/O contributes {att.get('Network I/O', '–')}% to anomaly score. "
            f"{'⚠ Elevated latency detected.' if mn['network'] > 70 else '✓ Network within bounds.'}"
        )
    if any(k in m for k in ["power", "watt", "energy", "draw", "psu"]):
        mp = max(racks, key=lambda x: x["power"])
        return (
            f"⚡ Highest power: <b>{mp['id']}</b> at <b>{mp['power']:.1f}%</b> capacity. "
            f"Power Draw contributes {att.get('Power Draw', '–')}% to anomaly score."
        )
    if any(k in m for k in ["risk", "fail", "critical", "alert", "anomaly", "danger", "status"]):
        cr = [r["id"] for r in racks if r["status"] == "critical"]
        wa = [r["id"] for r in racks if r["status"] == "warning"]
        return (
            f"🚨 Critical racks: {', '.join(cr) if cr else 'None'}. "
            f"Warning racks: {', '.join(wa) if wa else 'None'}. "
            f"Failure risk: <b>{scores['risk']}%</b>. Primary XAI driver: <b>{tf}</b> ({tp}%)."
        )
    if any(k in m for k in ["xai", "explain", "why", "cause", "attrib", "reason", "driver", "factor"]):
        lines = " | ".join([f"{k}:{v}%" for k, v in att.items()])
        return (
            f"🔬 XAI attribution — {lines}. "
            f"<b>{tf}</b> is dominant at {tp}%, meaning it correlates most strongly "
            f"with the anomaly score in this window."
        )
    if any(k in m for k in ["recomm", "action", "fix", "resol", "mitig", "what should", "what do", "suggest"]):
        return (
            f"💡 Top recommendation for {scenario}: Focus on <b>{tf}</b> (primary XAI driver). "
            f"Rack {h['id']} at {h['temp']}°C needs thermal attention. "
            f"Rack {b['id']} at {b['load']:.1f}% CPU — migrate if possible."
        )
    if any(k in m for k in ["health", "score", "overall", "summary"]):
        return (
            f"📊 System health: <b>{scores['health']}/100</b>. "
            f"Failure risk: <b>{scores['risk']}%</b>. Window: {scenario}. "
            f"{'Degraded.' if scores['risk'] > 70 else 'Stable.' if scores['risk'] < 35 else 'Moderate stress.'}"
        )

    rm = re.search(r"r[-\s]?0?([1-8])", m)
    if rm:
        rid = f"R-0{rm.group(1)}"
        r = next((x for x in racks if x["id"] == rid), None)
        if r:
            return (
                f"🗄️ <b>{r['id']}</b> ({r['zone']}) — "
                f"Temp:{r['temp']}°C | CPU:{r['load']:.0f}% | "
                f"Power:{r['power']:.0f}% | Net:{r['network']:.0f}ms | "
                f"Status:<b>{r['status'].upper()}</b> | Anomaly:{r['anomaly']}%"
            )

    return (
        "I can answer about: temperature, CPU, network, power, risk, XAI attribution, "
        "recommendations, or a specific rack (e.g. 'R-03'). What would you like to know?"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def chart_live_telemetry(df: pd.DataFrame):
    d = scale_df(df)
    d["lt"] = d["timestamp"].dt.strftime("%H:%M:%S")

    specs = [
        ("Temp (°C)", "temp_d", "#00e5ff", 18, 24),
        ("CPU Load (%)", "cpu_d", "#a855f7", 50, 75),
        ("Power (%)", "power_d", "#22c55e", 35, 90),
        ("Net (ms)", "net_d", "#fb7185", 0, 100),
    ]

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[1, 1, 1, 1],
    )

    for row, (lbl, col, color, ylo, yhi) in enumerate(specs, 1):
        y = d[col]
        fig.add_trace(
            go.Scatter(
                x=d["lt"], y=y, name=lbl, mode="lines",
                line=dict(color=color, width=2.0),
                hovertemplate=f"<b>{lbl}</b>: %{{y:.1f}}<extra></extra>"
            ),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[d["lt"].iloc[-1]], y=[float(y.iloc[-1])],
                mode="markers", showlegend=False,
                marker=dict(size=9, color=color, line=dict(color="white", width=1.5)),
                hoverinfo="skip"
            ),
            row=row, col=1
        )
        fig.update_yaxes(
            range=[ylo, yhi * 1.02],
            row=row, col=1,
            gridcolor="rgba(255,255,255,0.07)",
            color="#8aa3c3",
            tickfont=dict(family="Share Tech Mono", size=9),
            title_text=lbl,
            title_font=dict(size=9, color="#8aa3c3"),
            rangemode="normal"
        )

    fig.update_xaxes(
        showgrid=False, color="#8aa3c3",
        tickfont=dict(family="Share Tech Mono", size=9),
        nticks=10, row=4, col=1
    )
    fig.update_layout(
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified",
        font=dict(color="#7fa6c9", family="Exo 2"),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", font=dict(size=10, color="#d9ebff"))
    )
    return fig

def chart_alert_timeline(df: pd.DataFrame):
    d = scale_df(df)
    d["lt"] = d["timestamp"].dt.strftime("%H:%M:%S")

    fig = go.Figure()
    specs = [
        ("Temperature", "temp_d", 18, 24, "#00e5ff"),
        ("CPU Load", "cpu_d", 50, 75, "#a855f7"),
        ("Power", "power_d", 35, 90, "#22c55e"),
        ("Network", "net_d", 0, 100, "#fb7185"),
    ]

    for lbl, col, lo, hi, color in specs:
        s = d[col]
        normed = ((s - lo) / (hi - lo) * 100).clip(0, 100)
        r_, g_, b_ = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig.add_trace(go.Scatter(
            x=d["lt"], y=normed, name=lbl, mode="lines",
            line=dict(color=color, width=1.8),
            fill="tozeroy", fillcolor=f"rgba({r_},{g_},{b_},0.06)",
            hovertemplate=f"<b>{lbl}</b>: %{{y:.1f}}%<extra></extra>"
        ))

    anom = (df["anomaly_score"] * 100).clip(0, 100)
    fig.add_trace(go.Scatter(
        x=d["lt"], y=anom, name="Anomaly", mode="lines",
        line=dict(color="#ff2255", width=2.2, dash="dot"),
        hovertemplate="<b>Anomaly</b>: %{y:.1f}%<extra></extra>"
    ))

    fig.add_hline(
        y=60, line_dash="dash", line_color="rgba(255,180,0,0.55)",
        annotation_text="WARN 60%", annotation_font_color="#ffb400",
        annotation_font_size=9, annotation_font_family="Share Tech Mono",
        annotation_position="bottom right"
    )
    fig.add_hline(
        y=85, line_dash="dash", line_color="rgba(255,60,60,0.50)",
        annotation_text="CRIT 85%", annotation_font_color="#ff5555",
        annotation_font_size=9, annotation_font_family="Share Tech Mono",
        annotation_position="bottom right"
    )

    fig.update_layout(
        **_BASE, height=270, hovermode="x unified",
        font=dict(color="#7fa6c9", family="Exo 2"),
        xaxis=dict(showgrid=False, color="#8aa3c3", tickfont=dict(family="Share Tech Mono", size=10), nticks=10),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(255,255,255,0.07)",
            color="#8aa3c3", range=[0, 110],
            tickfont=dict(family="Share Tech Mono", size=10),
            ticksuffix="%"
        ),
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right", font=dict(size=10, color="#d9ebff"))
    )
    return fig

def chart_gauge(fp: int):
    color = "#ff3c3c" if fp > 70 else "#ffb400" if fp > 40 else "#00dc82"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=fp,
        number=dict(suffix="%", font=dict(size=32, color="white", family="Rajdhani")),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1,
                tickcolor="#8aa3c3",
                tickfont=dict(size=9, family="Share Tech Mono"),
                nticks=6
            ),
            bar=dict(color=color, thickness=0.25),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.06)",
            steps=[
                dict(range=[0, 40], color="rgba(0,220,130,0.10)"),
                dict(range=[40, 70], color="rgba(255,180,0,0.10)"),
                dict(range=[70, 100], color="rgba(255,60,60,0.12)")
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.7, value=fp),
        ),
        domain=dict(x=[0.05, 0.95], y=[0, 1])
    ))
    fig.update_layout(**_BASE, height=210, font=dict(color="#7fa6c9", family="Exo 2"))
    return fig

def chart_zone_bar(zd: dict, zf: str):
    zs = ["A", "B", "C", "D"] if zf == "All Zones" else [zf.replace("Zone ", "")]
    rs = [zd.get(z, 0) for z in zs]
    cs = ["#ff3c3c" if v > 70 else "#ffb400" if v > 40 else "#00dc82" for v in rs]

    fig = go.Figure(go.Bar(
        x=[f"Zone {z}" for z in zs], y=rs,
        marker=dict(color=cs),
        text=[f"{v}%" for v in rs],
        textposition="outside",
        textfont=dict(family="Share Tech Mono", size=11, color="white"),
        width=0.55
    ))
    fig.update_layout(
        **_BASE, height=230, showlegend=False,
        font=dict(color="#7fa6c9", family="Exo 2"),
        xaxis=dict(showgrid=False, color="#8aa3c3", tickfont=dict(family="Share Tech Mono", size=11)),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)",
                   color="#8aa3c3", range=[0, 115],
                   tickfont=dict(family="Share Tech Mono", size=10))
    )
    return fig

def chart_3d_racks(racks, zf):
    fil = racks if zf == "All Zones" else [r for r in racks if r["zone"] == zf]

    x, y, z, labels, hover_text, colors, sizes = [], [], [], [], [], [], []
    status_to_z = {"healthy": 1.0, "warning": 2.0, "critical": 3.0}

    for r in fil:
        px, py = RACK_POS[r["id"]]
        x.append(px)
        y.append(py)
        z.append(status_to_z[r["status"]])
        labels.append(r["id"])
        hover_text.append(
            f"<b>{r['id']}</b><br>{r['zone']}<br>"
            f"Temp: {r['temp']}°C<br>CPU: {r['load']:.0f}%<br>"
            f"Power: {r['power']:.0f}%<br>Net: {r['network']:.0f}ms<br>"
            f"Status: <b>{r['status'].upper()}</b>"
        )
        if r["status"] == "critical":
            colors.append("#ff3c3c")
            sizes.append(22)
        elif r["status"] == "warning":
            colors.append("#ffb400")
            sizes.append(17)
        else:
            colors.append("#00e5ff")
            sizes.append(13)

    fig = go.Figure()

    for gx in range(1, 5):
        fig.add_trace(go.Scatter3d(
            x=[gx, gx], y=[1, 2], z=[0, 0],
            mode="lines",
            line=dict(color="rgba(0,229,255,0.12)", width=3),
            hoverinfo="skip", showlegend=False
        ))
    for gy in range(1, 3):
        fig.add_trace(go.Scatter3d(
            x=[1, 4], y=[gy, gy], z=[0, 0],
            mode="lines",
            line=dict(color="rgba(0,229,255,0.12)", width=3),
            hoverinfo="skip", showlegend=False
        ))

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers+text",
        text=labels,
        hovertext=hover_text,
        hoverinfo="text",
        textposition="top center",
        marker=dict(size=sizes, color=colors, opacity=0.95, symbol="square", line=dict(color=colors, width=2)),
        textfont=dict(color="white", size=11, family="Share Tech Mono"),
        showlegend=False
    ))

    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Exo 2"),
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title=dict(text="Column (X →)", font=dict(size=10, color="#8aa3c3")),
                color="#8aa3c3",
                gridcolor="rgba(255,255,255,0.08)",
                showbackground=False,
                tickmode="array",
                tickvals=[1, 2, 3, 4],
                ticktext=["C1", "C2", "C3", "C4"],
                tickfont=dict(family="Share Tech Mono", size=10),
                range=[0.5, 4.5]
            ),
            yaxis=dict(
                title=dict(text="Row (Y →)", font=dict(size=10, color="#8aa3c3")),
                color="#8aa3c3",
                gridcolor="rgba(255,255,255,0.08)",
                showbackground=False,
                tickmode="array",
                tickvals=[1, 2],
                ticktext=["Row 1", "Row 2"],
                tickfont=dict(family="Share Tech Mono", size=10),
                range=[0.5, 2.5]
            ),
            zaxis=dict(
                title=dict(text="Risk (Z)", font=dict(size=10, color="#8aa3c3")),
                color="#8aa3c3",
                gridcolor="rgba(255,255,255,0.07)",
                showbackground=False,
                tickmode="array",
                tickvals=[0, 1, 2, 3],
                ticktext=["Base", "Low", "Warn", "Crit"],
                tickfont=dict(family="Share Tech Mono", size=10),
                range=[0, 3.4]
            ),
            camera=dict(eye=dict(x=1.55, y=1.35, z=1.0)),
            aspectmode="cube"
        ),
        uirevision="rack3d"
    )
    return fig

def chart_thermal_heatmap(racks, zf):
    fil = racks if zf == "All Zones" else [r for r in racks if r["zone"] == zf]
    rack_layout = {
        "R-01": (0, 1), "R-02": (1, 1), "R-03": (2, 1), "R-04": (3, 1),
        "R-05": (0, 0), "R-06": (1, 0), "R-07": (2, 0), "R-08": (3, 0),
    }

    fig = go.Figure()

    for r in fil:
        if r["id"] not in rack_layout:
            continue

        x0, y0 = rack_layout[r["id"]]
        x1, y1 = x0 + 1, y0 + 1
        box_color = get_rack_box_color(r["id"])

        fig.add_shape(
            type="rect",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color="rgba(255,255,255,0.85)", width=2),
            fillcolor=box_color,
            layer="below"
        )

        fig.add_annotation(
            x=x0 + 0.5, y=y0 + 0.5,
            text=f"<b>{r['id']}</b><br>{r['temp']}°C<br>{r['load']:.0f}% CPU",
            showarrow=False,
            font=dict(color="white", size=11, family="Share Tech Mono"),
            align="center"
        )

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,15,35,0.92)",
        font=dict(color="#d9ebff", family="Exo 2"),
        showlegend=False,
        xaxis=dict(
            range=[0, 4],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            tickmode="array",
            tickvals=[0.5, 1.5, 2.5, 3.5],
            ticktext=["Col 1", "Col 2", "Col 3", "Col 4"],
            color="#8aa3c3",
            title="Rack Columns",
            tickfont=dict(family="Share Tech Mono", size=10)
        ),
        yaxis=dict(
            range=[0, 2],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            tickmode="array",
            tickvals=[0.5, 1.5],
            ticktext=["Row 1", "Row 2"],
            color="#8aa3c3",
            title="Rack Rows",
            tickfont=dict(family="Share Tech Mono", size=10),
            scaleanchor="x",
            scaleratio=1
        )
    )
    return fig

def chart_risk_scatter(racks):
    """
    3D scatter with clearly visible points inside the plot.
    X = CPU Load % (50–75)
    Y = Temperature °C (18–24)
    Z = Power % (35–90)
    """
    fig = go.Figure()

    # grid lines for 3D box
    x_ticks = [50, 55, 60, 65, 70, 75]
    y_ticks = [18, 19, 20, 21, 22, 23, 24]
    z_ticks = [35, 50, 65, 80, 90]

    for xv in x_ticks:
        fig.add_trace(go.Scatter3d(
            x=[xv, xv], y=[18, 24], z=[35, 35],
            mode="lines", line=dict(color="rgba(230,240,255,0.20)", width=2),
            hoverinfo="skip", showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=[xv, xv], y=[18, 18], z=[35, 90],
            mode="lines", line=dict(color="rgba(230,240,255,0.16)", width=2),
            hoverinfo="skip", showlegend=False
        ))

    for yv in y_ticks:
        fig.add_trace(go.Scatter3d(
            x=[50, 75], y=[yv, yv], z=[35, 35],
            mode="lines", line=dict(color="rgba(230,240,255,0.20)", width=2),
            hoverinfo="skip", showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=[50, 50], y=[yv, yv], z=[35, 90],
            mode="lines", line=dict(color="rgba(230,240,255,0.16)", width=2),
            hoverinfo="skip", showlegend=False
        ))

    for zv in z_ticks:
        fig.add_trace(go.Scatter3d(
            x=[50, 75], y=[18, 18], z=[zv, zv],
            mode="lines", line=dict(color="rgba(230,240,255,0.18)", width=2),
            hoverinfo="skip", showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=[50, 50], y=[18, 24], z=[zv, zv],
            mode="lines", line=dict(color="rgba(230,240,255,0.14)", width=2),
            hoverinfo="skip", showlegend=False
        ))

    ids = [r["id"] for r in racks]
    x = [r["load"] for r in racks]
    y = [r["temp"] for r in racks]
    z = [r["power"] for r in racks]

    hover_text = [
        (
            f"<b>{r['id']}</b><br>"
            f"CPU Load: {r['load']:.1f}%<br>"
            f"Temperature: {r['temp']:.1f}°C<br>"
            f"Power: {r['power']:.1f}%<br>"
            f"Network: {r['network']:.1f} ms<br>"
            f"Anomaly: {r['anomaly']:.1f}%"
        )
        for r in racks
    ]

    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers+text",
        text=ids,
        textposition="top center",
        hovertext=hover_text,
        hoverinfo="text",
        marker=dict(
            size=10,
            color="#12e7ff",
            opacity=0.98,
            symbol="circle",
            line=dict(color="white", width=1.5)
        ),
        textfont=dict(color="white", size=12, family="Share Tech Mono"),
        showlegend=False
    ))

    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Exo 2"),
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            aspectmode="manual",
            aspectratio=dict(x=1.55, y=1.20, z=0.95),
            camera=dict(eye=dict(x=1.55, y=1.42, z=1.0)),
            xaxis=dict(
                title=dict(text="CPU Load %", font=dict(size=14, color="#a9c7e8")),
                showbackground=False,
                showgrid=False,
                zeroline=False,
                color="#a9c7e8",
                tickfont=dict(family="Share Tech Mono", size=10),
                range=[50, 75],
                tickmode="array",
                tickvals=x_ticks,
                ticktext=[str(v) for v in x_ticks],
            ),
            yaxis=dict(
                title=dict(text="Temperature °C", font=dict(size=14, color="#a9c7e8")),
                showbackground=False,
                showgrid=False,
                zeroline=False,
                color="#a9c7e8",
                tickfont=dict(family="Share Tech Mono", size=10),
                range=[18, 24],
                tickmode="array",
                tickvals=y_ticks,
                ticktext=[str(v) for v in y_ticks],
            ),
            zaxis=dict(
                title=dict(text="Power %", font=dict(size=14, color="#a9c7e8")),
                showbackground=False,
                showgrid=False,
                zeroline=False,
                color="#a9c7e8",
                tickfont=dict(family="Share Tech Mono", size=10),
                range=[35, 90],
                tickmode="array",
                tickvals=z_ticks,
                ticktext=[str(v) for v in z_ticks],
            )
        ),
        uirevision="risk_scatter_3d"
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
try:
    base_df, csv_path = load_base_dataframe()
    full_df = enrich(base_df)
except Exception as e:
    st.error(str(e))
    st.stop()

sw = scenario_windows(full_df)

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-REFRESH — always 2 sec
# ─────────────────────────────────────────────────────────────────────────────
LIVE_WIN = max(900, min(3600, len(full_df) // 8))
STEP = max(1, len(full_df) // 500)

if AUTO_REFRESH_AVAILABLE:
    st_autorefresh(interval=2000, key="live_refresh")
    st.session_state.live_pointer = (st.session_state.live_pointer + STEP) % len(full_df)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:0.6rem 0 1.4rem;'>
      <div style='font-size:40px;margin-bottom:6px;'>🛡️</div>
      <div style='font-family:"Rajdhani",sans-serif;font-size:22px;font-weight:700;color:white;letter-spacing:0.06em;'>AegisAI</div>
      <div style='font-family:"Share Tech Mono",monospace;font-size:9px;color:#8aa3c3;letter-spacing:0.2em;margin-top:2px;'>REAL TIME DATA CENTER VIEW</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<div style="font-family:\'Share Tech Mono\',monospace;font-size:9px;color:#8aa3c3;letter-spacing:0.16em;margin-bottom:8px;">⚙ DATA WINDOW</div>',
        unsafe_allow_html=True
    )
    st.session_state.scenario = st.selectbox(
        "Scenario Window",
        ["Latest / Live", "Stable Window", "Failure Window", "Recovery Window"]
    )

    st.markdown("---")
    st.markdown(
        '<div style="font-family:\'Share Tech Mono\',monospace;font-size:9px;color:#8aa3c3;letter-spacing:0.16em;margin-bottom:8px;">🎛 DISPLAY</div>',
        unsafe_allow_html=True
    )
    st.session_state.selected_zone = st.selectbox(
        "Zone Filter",
        ["All Zones", "Zone A", "Zone B", "Zone C", "Zone D"]
    )
    st.session_state.time_range = st.selectbox(
        "Time Range",
        ["Last 10 min", "Last 30 min", "Last 1 hr", "Last 6 hr"],
        index=1
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:9px;color:#8aa3c3;letter-spacing:0.16em;margin-bottom:6px;">📡 LIVE STATUS</div>
    <div style="background:rgba(0,220,130,0.10);border:1px solid rgba(0,220,130,0.25);
                border-radius:10px;padding:8px 12px;font-family:'Share Tech Mono',monospace;
                font-size:10px;color:#00dc82;">● STREAMING · 2s REFRESH</div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LIVE STATE
# ─────────────────────────────────────────────────────────────────────────────
scenario = st.session_state.scenario

if scenario == "Latest / Live":
    ptr = st.session_state.live_pointer
    ei = max(LIVE_WIN, ptr)
    si = max(0, ei - LIVE_WIN)
    if ptr + LIVE_WIN > len(full_df):
        ei = len(full_df)
        si = max(0, ei - LIVE_WIN)
    scenario_df = full_df.iloc[si:ei].reset_index(drop=True)
else:
    s_, e_ = sw[scenario]
    scenario_df = full_df.iloc[s_:e_].reset_index(drop=True)

df = slice_tr(scenario_df, st.session_state.time_range)
if len(df) < 2:
    df = scenario_df.tail(min(len(scenario_df), 50)).reset_index(drop=True)

zf = st.session_state.selected_zone
racks_ = build_racks(scenario_df)
sc = compute_scores(scenario_df, scenario)
att = xai(scenario_df)
zr = zone_risk(racks_)
ins = ai_insight(racks_, scenario, att)
recs = recommendations(racks_, scenario, att)

sd = scale_df(df)
prev_t = float(sd["temp_d"].iloc[-2]) if len(sd) > 1 else sc["lt"]
prev_c = float(sd["cpu_d"].iloc[-2]) if len(sd) > 1 else sc["lc"]
prev_p = float(sd["power_d"].iloc[-2]) if len(sd) > 1 else sc["lp"]
prev_n = float(sd["net_d"].iloc[-2]) if len(sd) > 1 else sc["ln"]

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
ts_now = pd.Timestamp.now().strftime("%Y-%m-%d  %H:%M:%S")
ldot = '<span style="color:#00dc82;font-size:14px;">●</span> LIVE' if scenario == "Latest / Live" else scenario.upper()

st.markdown(f"""
<div class="glass glass-cyan">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px;">
    <div style="display:flex;align-items:center;gap:16px;">
      <div style="width:56px;height:56px;border-radius:20px;display:flex;align-items:center;
                  justify-content:center;font-size:28px;
                  background:rgba(0,229,255,0.10);border:1px solid rgba(0,229,255,0.24);">🛡️</div>
      <div>
        <div style="font-family:'Rajdhani',sans-serif;font-size:28px;font-weight:700;color:white;letter-spacing:0.04em;">AegisAI</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#8aa3c3;letter-spacing:0.12em;margin-top:2px;">DATA CENTER COPILOT · {ts_now}</div>
      </div>
    </div>
    <div style="display:flex;align-items:center;flex-wrap:wrap;gap:6px;">
      <span class="pill {sc['pill']}">{ldot}</span>
      <span class="pill pill-info">DATASET · {len(full_df):,} ROWS</span>
      <span class="pill pill-info">{zf.upper()}</span>
      <span class="pill {'pill-crit' if sc['risk'] > 70 else 'pill-warn' if sc['risk'] > 40 else 'pill-ok'}">
        {'⚠ HIGH RISK' if sc['risk'] > 70 else '~ MODERATE' if sc['risk'] > 40 else '✓ NOMINAL'}
      </span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
kpis = [
    (k1, "SYSTEM HEALTH", f"{sc['health']}", "/100", "kpi-cyan",
     f'<span class="kpi-trend {"trend-down" if sc["risk"] > 70 else "trend-flat"}">{"▼ DEGRADED" if sc["risk"] > 70 else "— NOMINAL"}</span>'),
    (k2, "FAILURE RISK", f"{sc['risk']}", "%",
     "kpi-red" if sc["risk"] > 60 else "kpi-amber" if sc["risk"] > 35 else "kpi-green",
     f'<span class="kpi-trend {"trend-up" if sc["risk"] > 60 else "trend-down"}">{"▲ HIGH" if sc["risk"] > 60 else "▼ LOW"}</span>'),
    (k3, "TEMPERATURE", f"{sc['lt']:.1f}", "°C",
     "kpi-red" if sc["lt"] > 23 else "kpi-amber" if sc["lt"] > 21.5 else "kpi-cyan",
     delta_html(sc["lt"], prev_t)),
    (k4, "CPU LOAD", f"{sc['lc']:.1f}", "%",
     "kpi-red" if sc["lc"] > 70 else "kpi-amber" if sc["lc"] > 63 else "kpi-cyan",
     delta_html(sc["lc"], prev_c)),
    (k5, "POWER DRAW", f"{sc['lp']:.1f}", "%",
     "kpi-red" if sc["lp"] > 80 else "kpi-amber" if sc["lp"] > 65 else "kpi-cyan",
     delta_html(sc["lp"], prev_p)),
    (k6, "NETWORK", f"{sc['ln']:.1f}", "ms",
     "kpi-red" if sc["ln"] > 80 else "kpi-amber" if sc["ln"] > 50 else "kpi-cyan",
     delta_html(sc["ln"], prev_n)),
]

for col, label, val, unit, cls, trend in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi {cls}">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{val}<span class="kpi-unit">{unit}</span></div>
          {trend}
        </div>
        """, unsafe_allow_html=True)

st.markdown(f"""
<div style="background:{'rgba(255,60,60,0.08)' if sc['risk'] > 70 else 'rgba(0,220,130,0.06)' if sc['risk'] < 35 else 'rgba(255,180,0,0.08)'};
            border:1px solid {'rgba(255,60,60,0.34)' if sc['risk'] > 70 else 'rgba(0,220,130,0.26)' if sc['risk'] < 35 else 'rgba(255,180,0,0.26)'};
            border-radius:16px;padding:12px 18px;margin:14px 0;font-size:13px;line-height:1.7;color:#e0f0ff;">
  {ins}
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 1 — Live chart + Right panel
# ─────────────────────────────────────────────────────────────────────────────
L, R = st.columns([1.65, 1.0], gap="large")

with L:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    lbl_ = '<span style="color:#00dc82;">●</span> Live Feed' if scenario == "Latest / Live" else "Telemetry"
    st.markdown(f'<div class="sec-title">{lbl_} — {st.session_state.time_range}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sec-sub">SOURCE › {os.path.basename(csv_path).upper()} · {scenario.upper()}</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_live_telemetry(df), use_container_width=True, config={"displayModeBar": False}, key="live_chart")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Alert Timeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">METRICS NORMALISED 0–100% · ANOMALY OVERLAY (dotted red) · WARN=60 CRIT=85</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_alert_timeline(df), use_container_width=True, config={"displayModeBar": False}, key="alert_tl")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown(f'<div class="sec-title">Rack Risk Elevation — {zf}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">X=COLUMN · Y=ROW · Z=RISK LEVEL · RED=CRIT · AMBER=WARN · CYAN=OK</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_3d_racks(racks_, zf), use_container_width=True, config={"displayModeBar": False}, key="rack_3d")
    st.markdown('</div>', unsafe_allow_html=True)

with R:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Failure Risk</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_gauge(sc["fp"]), use_container_width=True, config={"displayModeBar": False}, key="gauge")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">XAI — Anomaly Attribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">CORRELATION-WEIGHTED FEATURE CONTRIBUTION</div>', unsafe_allow_html=True)
    st.markdown(xai_html(att), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown(f'<div class="sec-title">Zone Risk — {zf}</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_zone_bar(zr, zf), use_container_width=True, config={"displayModeBar": False}, key="zone_risk")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">AI Recommendations</div>', unsafe_allow_html=True)
    if recs:
        for prio, pc, pbg, title, desc in recs:
            st.markdown(f"""
            <div class="rec">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:3px;">
                <span style="background:{pbg};color:{pc};border:1px solid {pc}40;border-radius:999px;
                             padding:2px 9px;font-family:'Share Tech Mono',monospace;font-size:9px;
                             letter-spacing:0.1em;">{prio}</span>
                <span class="rec-title">{title}</span>
              </div>
              <div class="rec-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="ai-insight">✦ System within nominal parameters.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 2 — Thermal Heatmap | Risk Scatter
# ─────────────────────────────────────────────────────────────────────────────
h1, h2 = st.columns([1.3, 1.0], gap="large")

with h1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown(f'<div class="sec-title">Thermal Heat Map — {scenario}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">ONE RACK = ONE BOX = ONE COLOR · BLUE / GREEN / YELLOW / RED</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_thermal_heatmap(racks_, zf), use_container_width=True, config={"displayModeBar": False}, key="heatmap")
    st.markdown('</div>', unsafe_allow_html=True)

with h2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">3D Rack Risk Scatter</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">X=CPU 50–75% · Y=TEMPERATURE °C · Z=POWER % · CYAN=RACK NODE</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_risk_scatter(racks_), use_container_width=True, config={"displayModeBar": False}, key="scatter")
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 3 — What-If Simulation
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="sec-title">What-If Simulation</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-sub">PROJECTED IMPACT OF RECOMMENDED ACTIONS · XAI ATTRIBUTED</div>', unsafe_allow_html=True)

wif = build_what_if(sc, racks_, scenario, att)
sc1, sc2, sc3 = st.columns([1, 1, 2])

with sc1:
    st.markdown(f"""
    <div style="background:rgba(255,60,60,0.08);border:1px solid rgba(255,60,60,0.22);border-radius:16px;padding:16px 18px;">
      <div style="font-family:'Share Tech Mono',monospace;font-size:9px;color:#ff8a8a;letter-spacing:0.18em;margin-bottom:6px;">BEFORE FIX</div>
      <div style="font-family:'Rajdhani',sans-serif;font-size:38px;font-weight:700;color:white;line-height:1;">{wif['before_risk']}%</div>
      <div style="color:#ffd0d0;font-size:11px;margin-top:4px;">Current failure risk</div>
      <div style="color:#ffd0d0;font-size:11px;margin-top:2px;">Health: <b>{wif['before_health']}</b>/100</div>
    </div>
    """, unsafe_allow_html=True)

with sc2:
    st.markdown(f"""
    <div style="background:rgba(0,220,130,0.08);border:1px solid rgba(0,220,130,0.22);border-radius:16px;padding:16px 18px;">
      <div style="font-family:'Share Tech Mono',monospace;font-size:9px;color:#00ff9a;letter-spacing:0.18em;margin-bottom:6px;">AFTER FIX</div>
      <div style="font-family:'Rajdhani',sans-serif;font-size:38px;font-weight:700;color:white;line-height:1;">{wif['after_risk']}%</div>
      <div style="color:#d9ffee;font-size:11px;margin-top:4px;">Projected risk after actions</div>
      <div style="color:#d9ffee;font-size:11px;margin-top:2px;">Health: <b>{wif['after_health']}</b>/100</div>
    </div>
    """, unsafe_allow_html=True)

with sc3:
    st.markdown(f"""
    <div class="ai-insight">
      Actions project risk <b>{wif['before_risk']}%</b> → <b>{wif['after_risk']}%</b>,
      health <b>{wif['before_health']}</b> → <b>{wif['after_health']}</b>.
      XAI primary driver: <b>{wif['top_feat']}</b>.
      Hottest: <b>{wif['hottest_rack']}</b> at <b>{wif['hottest_temp']}°C</b>.
    </div>
    """, unsafe_allow_html=True)

if st.button("🔍 Run Full What-If Analysis", use_container_width=True):
    st.session_state.what_if_ran = True
    st.session_state.what_if_report = build_what_if(sc, racks_, scenario, att)

if st.session_state.what_if_ran and st.session_state.what_if_report:
    rpt = st.session_state.what_if_report
    st.markdown('<div class="sec-sub" style="margin-top:10px;">RECOVERY STEPS</div>', unsafe_allow_html=True)
    cols_ = st.columns(len(rpt["actions"]))
    for i, (c_, a_) in enumerate(zip(cols_, rpt["actions"]), 1):
        with c_:
            st.markdown(f'<div class="rec"><div class="rec-title">Step {i}</div><div class="rec-desc">{a_}</div></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 4 — AI Chatbot
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="glass glass-cyan">', unsafe_allow_html=True)
st.markdown('<div class="sec-title">🤖 AegisAI Copilot — Chat</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-sub">ASK ABOUT TEMPERATURE · CPU · NETWORK · RISK · RACKS · XAI · RECOMMENDATIONS</div>', unsafe_allow_html=True)

ch = st.session_state.chat_history
chat_html = '<div class="chat-wrap">'
for m in ch:
    cls_ = "chat-user" if m["role"] == "user" else "chat-bot"
    icon = "You" if m["role"] == "user" else "🛡️ AegisAI"
    align = "flex-end" if m["role"] == "user" else "flex-start"
    chat_html += (
        f'<div style="display:flex;flex-direction:column;align-items:{align};">'
        f'<div class="chat-bubble {cls_}">{m["text"]}</div>'
        f'<div class="chat-ts">{icon} · {m["ts"]}</div></div>'
    )
if not ch:
    chat_html += '<div style="color:#8aa3c3;font-size:12px;font-family:\'Share Tech Mono\',monospace;padding:10px;">Type a question below to start…</div>'
chat_html += '</div>'
st.markdown(chat_html, unsafe_allow_html=True)

ci1, ci2 = st.columns([5, 1])
with ci1:
    user_input = st.text_input(
        "",
        placeholder="e.g. What is the hottest rack? | Why is risk high? | R-03 status",
        key="chat_input",
        label_visibility="collapsed"
    )
with ci2:
    send = st.button("Send ➤", use_container_width=True, key="chat_send")

if send and user_input.strip():
    ts_c = pd.Timestamp.now().strftime("%H:%M:%S")
    st.session_state.chat_history.append({"role": "user", "text": user_input.strip(), "ts": ts_c})
    reply = chatbot(user_input.strip(), sc, racks_, att, scenario)
    st.session_state.chat_history.append({"role": "bot", "text": reply, "ts": ts_c})
    st.rerun()

qcols = st.columns(4)
quick = ["What's the hottest rack?", "Why is risk high?", "Show XAI breakdown", "What to do now?"]
for qc, q in zip(qcols, quick):
    with qc:
        if st.button(q, key=f"q_{q}", use_container_width=True):
            ts_c = pd.Timestamp.now().strftime("%H:%M:%S")
            st.session_state.chat_history.append({"role": "user", "text": q, "ts": ts_c})
            reply = chatbot(q, sc, racks_, att, scenario)
            st.session_state.chat_history.append({"role": "bot", "text": reply, "ts": ts_c})
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# RAW DATA
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("🗃 Raw Telemetry Data (display-scaled)"):
    ds = scale_df(df)
    st.dataframe(
        ds[["timestamp", "temp_d", "cpu_d", "power_d", "net_d", "anomaly_score"]]
        .rename(columns={
            "temp_d": "temp_°C",
            "cpu_d": "cpu_%",
            "power_d": "power_%",
            "net_d": "network_ms",
        })
        .copy(),
        use_container_width=True
    )

st.markdown("""
<div style="text-align:center;padding:1.2rem 0 0.4rem;font-family:'Share Tech Mono',monospace;
            font-size:9px;letter-spacing:0.22em;color:#8aa3c3;">
  AEGISAI · 2s AUTO-REFRESH · XAI · CHATBOT · DATASET DRIVEN
</div>
""", unsafe_allow_html=True)
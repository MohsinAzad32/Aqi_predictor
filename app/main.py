"""
app/main.py — Multan AQI Predictor Dashboard
Professional 3-day forecast dashboard with SHAP explainability
Real-time predictions from Google Cloud Vertex AI Model Registry
"""

import json, pickle, warnings
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multan AQI Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS  — refined dark theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
    background-color: #080d17;
    color: #e2e8f0;
}
.stApp { background-color: #080d17; }
.block-container { padding: 1.5rem 2.5rem 2rem; }
#MainMenu, footer, header { visibility: hidden; }

.aqi-card {
    background: linear-gradient(135deg, #0f1929 0%, #111d30 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 28px 32px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.metric-card {
    background: #0f1929;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 22px 20px;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }

.page-title {
    font-family: 'Manrope', sans-serif;
    font-size: 2rem; font-weight: 800;
    letter-spacing: -0.5px; color: #f1f5f9; margin: 0;
}
.page-sub { font-size: 0.95rem; color: #64748b; margin: 4px 0 0; font-weight: 300; }
.section-title {
    font-family: 'Manrope', sans-serif;
    font-size: 1.1rem; font-weight: 700;
    letter-spacing: 0.3px; color: #cbd5e1; margin-bottom: 4px;
}
.metric-label {
    font-size: 0.78rem; font-weight: 500; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px;
}
.metric-value {
    font-family: 'Manrope', sans-serif;
    font-size: 2.4rem; font-weight: 800; line-height: 1;
}
.alert-box {
    border-radius: 12px; padding: 18px 22px;
    border-left: 4px solid; margin-top: 4px;
}
.alert-msg { font-size: 0.92rem; line-height: 1.6; color: #e2e8f0; margin: 6px 0 0; }
.day-card {
    background: #0f1929; border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; padding: 20px; text-align: center;
}
.day-label {
    font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1px; color: #475569; margin-bottom: 10px;
}
.day-aqi { font-family: 'Manrope', sans-serif; font-size: 2.8rem; font-weight: 800; line-height: 1; }
.day-cat {
    font-size: 0.78rem; font-weight: 500; margin-top: 6px;
    padding: 3px 10px; border-radius: 20px; display: inline-block;
}
.range-row { display: flex; justify-content: center; gap: 16px; margin-top: 10px; }
.range-item { font-size: 0.8rem; color: #64748b; }
.range-val  { font-weight: 600; color: #94a3b8; }
.poll-row { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
.poll-name { font-size: 0.82rem; font-weight: 600; color: #94a3b8; width: 52px; }
.poll-bar-bg { flex: 1; height: 8px; background: #1e293b; border-radius: 4px; overflow: hidden; }
.poll-bar-fill { height: 100%; border-radius: 4px; }
.poll-val { font-size: 0.82rem; font-weight: 600; color: #cbd5e1; width: 68px; text-align: right; }
.hdivider { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 24px 0; }
.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    border-radius: 20px; padding: 4px 12px;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.4px;
}
.status-live {
    background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.25); color: #10b981;
}
.status-fallback {
    background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.25); color: #f59e0b;
}
.dot { width: 6px; height: 6px; border-radius: 50%; animation: pulse 2s infinite; }
.dot-green  { background: #10b981; }
.dot-amber  { background: #f59e0b; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
.legend-row { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px; }
.legend-item { display:flex; align-items:center; gap:5px; font-size:0.72rem; color:#64748b; }
.legend-dot { width:8px; height:8px; border-radius:2px; flex-shrink:0; }

/* Source badge */
.source-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(56,189,248,0.10); border: 1px solid rgba(56,189,248,0.2);
    border-radius: 8px; padding: 3px 10px;
    font-size: 0.7rem; font-weight: 600; color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
DATA_CSV  = ROOT / "data" / "multan_features.csv"

# ─────────────────────────────────────────────────────────────
# GCP CONFIGURATION  — edit these to match your project
# ─────────────────────────────────────────────────────────────
GCP_PROJECT    = "pearl-aqi-predictor"
GCP_REGION     = "us-central1"
ENDPOINT_NAME  = "aqi-predictor-endpoint"   # Your Vertex AI endpoint display name
BQ_TABLE       = f"{GCP_PROJECT}.aqi_feature_store.multan_historical"

# ─────────────────────────────────────────────────────────────
# EPA AQI SCALE
# ─────────────────────────────────────────────────────────────
AQI_SCALE = [
    (50,  "Good",                          "#00E400", "rgba(0,228,0,0.12)",
     "🟢", "Air quality is satisfactory and poses little or no risk. Enjoy outdoor activities freely."),
    (100, "Moderate",                      "#D4B800", "rgba(255,215,0,0.12)",
     "🟡", "Air quality is acceptable. Unusually sensitive individuals should consider reducing prolonged outdoor exertion."),
    (150, "Unhealthy for Sensitive Groups","#FF7E00", "rgba(255,126,0,0.12)",
     "🟠", "Sensitive groups may experience health effects. The general public is not likely to be affected."),
    (200, "Unhealthy",                     "#FF0000", "rgba(255,0,0,0.12)",
     "🔴", "Everyone may begin to experience health effects. Limit prolonged outdoor exertion."),
    (300, "Very Unhealthy",                "#8F3F97", "rgba(143,63,151,0.12)",
     "🟣", "Health alert — everyone may experience serious health effects. Wear N95 masks if going outside."),
    (999, "Hazardous",                     "#7E0023", "rgba(126,0,35,0.12)",
     "⛔", "🚨 Emergency conditions. Stay indoors, seal windows, use air purifiers."),
]

def get_aqi_info(val):
    val = max(0, int(val or 0))
    for limit, cat, color, bg, icon, msg in AQI_SCALE:
        if val <= limit:
            return dict(value=val, category=cat, color=color, bg=bg, icon=icon, message=msg)
    return dict(value=val, category="Hazardous", color="#7E0023", bg="rgba(126,0,35,0.12)",
                icon="⛔", message=AQI_SCALE[-1][5])

AQI_PLOTLY_ZONES = [
    dict(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=0,   y1=50,  fillcolor="#00E400", opacity=0.06, layer="below", line_width=0),
    dict(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=50,  y1=100, fillcolor="#D4B800", opacity=0.06, layer="below", line_width=0),
    dict(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=100, y1=150, fillcolor="#FF7E00", opacity=0.06, layer="below", line_width=0),
    dict(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=150, y1=200, fillcolor="#FF0000", opacity=0.07, layer="below", line_width=0),
    dict(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=200, y1=300, fillcolor="#8F3F97", opacity=0.06, layer="below", line_width=0),
    dict(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=300, y1=500, fillcolor="#7E0023", opacity=0.06, layer="below", line_width=0),
]

# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (matches training notebook exactly)
# ─────────────────────────────────────────────────────────────
def build_features_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    for lag in [1, 2, 3, 6, 12, 24, 48, 72]:
        df[f"aqi_lag_{lag}h"] = df["aqi"].shift(lag)
    for lag in [1, 2, 3]:
        df[f"pm2_5_lag_{lag}h"] = df["pm2_5"].shift(lag)
        df[f"co_lag_{lag}h"]    = df["co"].shift(lag)
    for win in [3, 6, 12, 24]:
        df[f"aqi_roll_mean_{win}h"] = df["aqi"].rolling(win).mean()
        df[f"aqi_roll_std_{win}h"]  = df["aqi"].rolling(win).std().fillna(0)
    df["aqi_rate_1h"]   = df["aqi"].diff(1)
    df["aqi_rate_3h"]   = df["aqi"].diff(3)
    df["aqi_trend_dir"] = (df["aqi_rate_1h"] > 0).astype(int)
    df["aqi_accel"]     = df["aqi_rate_1h"].diff(1)
    df["hour"]          = df["timestamp"].dt.hour
    df["day_of_week"]   = df["timestamp"].dt.dayofweek
    df["month"]         = df["timestamp"].dt.month
    df["day_of_year"]   = df["timestamp"].dt.dayofyear
    df["hour_sin"]      = np.sin(2 * np.pi * df["hour"]        / 24)
    df["hour_cos"]      = np.cos(2 * np.pi * df["hour"]        / 24)
    df["dow_sin"]       = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]       = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"]     = np.sin(2 * np.pi * df["month"]       / 12)
    df["month_cos"]     = np.cos(2 * np.pi * df["month"]       / 12)
    df["doy_sin"]       = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]       = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["is_morning_peak"] = df["hour"].isin([7, 8, 9, 10]).astype(int)
    df["is_evening_peak"] = df["hour"].isin([17, 18, 19, 20]).astype(int)
    def _season(m):
        if m in [12, 1, 2]: return 0
        elif m in [3, 4, 5]: return 1
        elif m in [6, 7, 8]: return 2
        return 3
    df["season"] = df["month"].apply(_season)
    return df.ffill().bfill().dropna().reset_index(drop=True)

# ─────────────────────────────────────────────────────────────
# VERTEX AI ONLINE PREDICTION
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_vertex_endpoint():
    """
    Resolves the Vertex AI endpoint for online prediction.
    Returns (endpoint_client, endpoint_resource_name) or (None, None).

    This connects to your deployed model in the Vertex AI Model Registry.
    The endpoint must be deployed at:
      https://console.cloud.google.com/vertex-ai/endpoints
    """
    try:
        from google.cloud import aiplatform
        aiplatform.init(project=GCP_PROJECT, location=GCP_REGION)
        endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{ENDPOINT_NAME}"',
            order_by="create_time desc",
        )
        if not endpoints:
            return None, None
        ep = endpoints[0]
        return ep, ep.resource_name
    except Exception as e:
        return None, str(e)


def predict_via_vertex(endpoint, feature_rows: list[dict], feature_cols: list[str]) -> list[float]:
    """
    Sends a batch of feature rows to the Vertex AI endpoint and returns predictions.

    Args:
        endpoint:      aiplatform.Endpoint object
        feature_rows:  list of dicts, one per timestep
        feature_cols:  ordered list of feature names matching training order

    Returns:
        list of float AQI predictions
    """
    # Build instances in the format Vertex AI expects (list of dicts or list of lists)
    instances = []
    for row in feature_rows:
        instance = {col: float(row.get(col, 0.0)) for col in feature_cols}
        instances.append(instance)

    response = endpoint.predict(instances=instances)
    # Vertex AI returns predictions as a list; handle both scalar and nested list
    preds = []
    for p in response.predictions:
        if isinstance(p, (list, tuple)):
            preds.append(float(p[0]))
        else:
            preds.append(float(p))
    return preds


# ─────────────────────────────────────────────────────────────
# LOCAL MODEL FALLBACK (artifacts/*.pkl)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_local_artifacts():
    """
    Loads the locally saved XGBoost model and scaler from the artifacts/ directory.
    Used as a fallback when Vertex AI is unavailable.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model = pickle.load(open(ARTIFACTS / "aqi_best_model.pkl", "rb"))

    sl = ARTIFACTS / "aqi_scaler.joblib"
    scaler = joblib.load(sl if sl.exists() else ARTIFACTS / "aqi_scaler.pkl")

    fp = ARTIFACTS / "feature_cols.json"
    if fp.exists():
        with open(fp) as f:
            raw = json.load(f)
        feature_cols = raw if isinstance(raw, list) else raw["features"]
    else:
        feature_cols = [
            "pm2_5","pm10","no2","so2","o3","co",
            "aqi_lag_1h","aqi_lag_2h","aqi_lag_3h","aqi_lag_6h",
            "aqi_lag_12h","aqi_lag_24h","aqi_lag_48h","aqi_lag_72h",
            "pm2_5_lag_1h","pm2_5_lag_2h","pm2_5_lag_3h",
            "co_lag_1h","co_lag_2h","co_lag_3h",
            "aqi_roll_mean_3h","aqi_roll_std_3h","aqi_roll_mean_6h","aqi_roll_std_6h",
            "aqi_roll_mean_12h","aqi_roll_std_12h","aqi_roll_mean_24h","aqi_roll_std_24h",
            "aqi_rate_1h","aqi_rate_3h","aqi_trend_dir","aqi_accel",
            "hour","day_of_week","month","day_of_year",
            "hour_sin","hour_cos","dow_sin","dow_cos",
            "month_sin","month_cos","doy_sin","doy_cos",
            "is_morning_peak","is_evening_peak","season",
        ]
    return model, scaler, feature_cols


# ─────────────────────────────────────────────────────────────
# UNIFIED MODEL LOADER
# Returns (predict_fn, feature_cols, model_source_label)
# predict_fn(feature_rows) -> list[float]
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def resolve_model():
    """
    Priority order:
      1. Vertex AI deployed endpoint (real-time, always the latest trained model)
      2. Local artifact file (aqi_best_model.pkl) — offline fallback

    Returns:
        predict_fn    : callable(feature_rows: list[dict]) -> list[float]
        feature_cols  : list[str]
        source_label  : str shown in UI badge
    """
    endpoint, ep_name = get_vertex_endpoint()

    if endpoint is not None:
        _, scaler, feature_cols = load_local_artifacts()

        def vertex_predict(feature_rows: list[dict]) -> list[float]:
            """Scale features locally, then call Vertex AI for inference."""
            df_in = pd.DataFrame(feature_rows)
            for col in feature_cols:
                if col not in df_in.columns:
                    df_in[col] = 0.0
            df_in = df_in[feature_cols]
            scaled = scaler.transform(df_in)
            scaled_rows = [
                {col: float(scaled[i, j]) for j, col in enumerate(feature_cols)}
                for i in range(len(scaled))
            ]
            return predict_via_vertex(endpoint, scaled_rows, feature_cols)

        return vertex_predict, feature_cols, "Vertex AI (Live)"

    # Fallback: local pkl model
    model, scaler, feature_cols = load_local_artifacts()

    def local_predict(feature_rows: list[dict]) -> list[float]:
        df_in = pd.DataFrame(feature_rows)
        for col in feature_cols:
            if col not in df_in.columns:
                df_in[col] = 0.0
        df_in = df_in[feature_cols]
        return list(model.predict(scaler.transform(df_in)).astype(float))

    return local_predict, feature_cols, "Local Artifact (Offline)"


# ─────────────────────────────────────────────────────────────
# DATA SOURCE  (BigQuery real-time → local CSV → demo)
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def load_data():
    """
    Attempts to load the most recent 500 rows of air quality data:
      1. BigQuery (real-time pipeline data)
      2. Local CSV  (last exported snapshot)
      3. Synthetic demo data  (realistic Multan values)

    The timestamp shown in the UI reflects the ACTUAL last data point.
    """
    # ── 1. BigQuery ──────────────────────────────────────────
    try:
        from google.cloud import bigquery
        import google.auth
        creds, _ = google.auth.default()
        client = bigquery.Client(project=GCP_PROJECT, credentials=creds)
        q = f"""
            SELECT CAST(timestamp AS TIMESTAMP) AS timestamp,
                   aqi, pm2_5, pm10, no2, so2, o3, co
            FROM `{BQ_TABLE}`
            ORDER BY timestamp DESC LIMIT 500
        """
        df = client.query(q).to_dataframe()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp").reset_index(drop=True), "BigQuery (Real-Time)"
    except Exception:
        pass

    # ── 2. Local CSV ─────────────────────────────────────────
    if DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.to_datetime(
                df.get("dt", df.index),
                unit="s" if "dt" in df.columns else None
            )
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        needed = ["timestamp","aqi","pm2_5","pm10","no2","so2","o3","co"]
        df = df[[c for c in needed if c in df.columns]]
        return df.sort_values("timestamp").reset_index(drop=True), "Local CSV (Cached)"

    # ── 3. Demo data ─────────────────────────────────────────
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    timestamps = [now - timedelta(hours=i) for i in range(199, -1, -1)]
    np.random.seed(42)
    base = 165 + 15*np.sin(np.linspace(0, 4*np.pi, 200)) + np.random.normal(0, 8, 200)
    df = pd.DataFrame({
        "timestamp": timestamps,
        "aqi":   np.clip(base, 80, 280).astype(int),
        "pm2_5": np.clip(base*0.45 + np.random.normal(0,3,200), 20, 150),
        "pm10":  np.clip(base*0.90 + np.random.normal(0,5,200), 40, 300),
        "no2":   np.clip(30 + np.random.normal(0,4,200), 10, 80),
        "so2":   np.clip(14 + np.random.normal(0,2,200), 4,  40),
        "o3":    np.clip(48 + np.random.normal(0,5,200), 20, 100),
        "co":    np.clip(1.4 + np.random.normal(0,0.1,200), 0.5, 4.0),
    })
    return df, "Demo (No Connection)"


# ─────────────────────────────────────────────────────────────
# 72-HOUR RECURSIVE FORECAST
# ─────────────────────────────────────────────────────────────
def generate_forecast(raw_df, predict_fn, feature_cols, hours=72):
    """
    Generates a 72-step recursive AQI forecast using the resolved model
    (either Vertex AI endpoint or local artifact).

    Each predicted step feeds back into the next step's lag features,
    enabling multi-step-ahead forecasting without a retraining loop.
    """
    tail   = raw_df.sort_values("timestamp").tail(200)
    aqi_q  = deque(tail["aqi"].tolist(),   maxlen=200)
    pm25_q = deque(tail["pm2_5"].tolist(), maxlen=200)
    co_q   = deque(tail["co"].tolist(),    maxlen=200)
    last_row = tail.iloc[-1]
    last_ts  = pd.Timestamp(last_row["timestamp"])
    forecasts = []
    batch_rows = []

    for h in range(1, hours + 1):
        ts = last_ts + timedelta(hours=h)
        a  = list(aqi_q);  p = list(pm25_q);  c = list(co_q)
        row = {
            "pm2_5": p[-1], "pm10": float(last_row.get("pm10", 0)),
            "no2":   float(last_row.get("no2", 0)), "so2": float(last_row.get("so2", 0)),
            "o3":    float(last_row.get("o3", 0)),  "co":  c[-1],
        }
        for lag in [1, 2, 3, 6, 12, 24, 48, 72]:
            row[f"aqi_lag_{lag}h"] = a[-lag] if len(a) >= lag else a[0]
        for lag in [1, 2, 3]:
            row[f"pm2_5_lag_{lag}h"] = p[-lag] if len(p) >= lag else p[0]
            row[f"co_lag_{lag}h"]    = c[-lag] if len(c) >= lag else c[0]
        for win in [3, 6, 12, 24]:
            w = a[-win:] if len(a) >= win else a
            row[f"aqi_roll_mean_{win}h"] = float(np.mean(w))
            row[f"aqi_roll_std_{win}h"]  = float(np.std(w)) if len(w) > 1 else 0.0
        row["aqi_rate_1h"]   = a[-1] - a[-2] if len(a) >= 2 else 0
        row["aqi_rate_3h"]   = a[-1] - a[-4] if len(a) >= 4 else 0
        row["aqi_trend_dir"] = 1 if row["aqi_rate_1h"] > 0 else 0
        row["aqi_accel"]     = (row["aqi_rate_1h"] - (a[-2]-a[-3])) if len(a) >= 3 else 0
        row["hour"]          = ts.hour;   row["day_of_week"] = ts.dayofweek
        row["month"]         = ts.month;  row["day_of_year"] = ts.dayofyear
        row["hour_sin"]      = np.sin(2*np.pi*ts.hour/24)
        row["hour_cos"]      = np.cos(2*np.pi*ts.hour/24)
        row["dow_sin"]       = np.sin(2*np.pi*ts.dayofweek/7)
        row["dow_cos"]       = np.cos(2*np.pi*ts.dayofweek/7)
        row["month_sin"]     = np.sin(2*np.pi*ts.month/12)
        row["month_cos"]     = np.cos(2*np.pi*ts.month/12)
        row["doy_sin"]       = np.sin(2*np.pi*ts.dayofyear/365)
        row["doy_cos"]       = np.cos(2*np.pi*ts.dayofyear/365)
        row["is_morning_peak"] = 1 if ts.hour in [7,8,9,10]   else 0
        row["is_evening_peak"] = 1 if ts.hour in [17,18,19,20] else 0
        m = ts.month
        row["season"] = 0 if m in [12,1,2] else 1 if m in [3,4,5] else 2 if m in [6,7,8] else 3

        # Predict one step at a time (must be recursive)
        pred = float(predict_fn([row])[0])
        pred = max(0.0, min(pred, 500.0))
        forecasts.append({"timestamp": ts, "aqi": pred})
        aqi_q.append(pred); pm25_q.append(p[-1]); co_q.append(c[-1])

    return pd.DataFrame(forecasts)


# ─────────────────────────────────────────────────────────────
# SHAP COMPUTATION  (always uses local model for explainability)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_shap(_model, _scaler, _feature_cols, _raw_df):
    shap_img = ARTIFACTS / "shap_summary.png"
    if shap_img.exists():
        import time
        age_h = (time.time() - shap_img.stat().st_mtime) / 3600
        if age_h < 24:
            return str(shap_img), None, None
    try:
        df_feat = build_features_df(
            _raw_df[["timestamp","aqi","pm2_5","pm10","no2","so2","o3","co"]].tail(300)
        ).tail(100)
        X = df_feat[[c for c in _feature_cols if c in df_feat.columns]]
        for c in _feature_cols:
            if c not in X.columns: X[c] = 0.0
        X = X[_feature_cols]
        Xs = pd.DataFrame(_scaler.transform(X), columns=_feature_cols)
        explainer   = shap.Explainer(_model, Xs)
        shap_values = explainer(Xs)
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#0f1929"); ax.set_facecolor("#0f1929")
        shap.summary_plot(shap_values, Xs, show=False, plot_size=None)
        ax = plt.gca(); ax.set_facecolor("#0f1929")
        ax.tick_params(colors="#94a3b8", labelsize=9)
        ax.xaxis.label.set_color("#94a3b8")
        plt.tight_layout()
        out = ARTIFACTS / "shap_summary_live.png"
        plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="#0f1929")
        plt.close()
        mean_shap = np.abs(shap_values.values).mean(axis=0)
        top_idx   = np.argsort(mean_shap)[::-1][:8]
        top_feats = [(str(_feature_cols[i]), float(mean_shap[i])) for i in top_idx]
        return str(out), shap_values, top_feats
    except Exception:
        return None, None, None


# ─────────────────────────────────────────────────────────────
# HELPER — Plotly dark layout
# ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0a1020",
    font=dict(family="Manrope, sans-serif", color="#94a3b8", size=11),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.06)", borderwidth=1),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.08)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.08)", zeroline=False),
)

def pl(**overrides):
    merged = {**PLOTLY_LAYOUT}
    merged.update(overrides)
    return merged


# ─────────────────────────────────────────────────────────────
# ███  MAIN APP  ███
# ─────────────────────────────────────────────────────────────
with st.spinner(""):
    predict_fn, feature_cols, model_source = resolve_model()
    raw_df, data_source = load_data()
    local_model, local_scaler, _ = load_local_artifacts()

# Current values from latest data point
latest       = raw_df.iloc[-1]
current_aqi  = int(latest["aqi"])
current_info = get_aqi_info(current_aqi)
# ── KEY FIX: always show the ACTUAL timestamp from the data ──
last_updated = pd.to_datetime(latest["timestamp"]).strftime("%d %b %Y, %H:%M")

pollutants = {
    "PM2.5": (float(latest.get("pm2_5", 0)), 75,  "μg/m³", "#f59e0b"),
    "PM10" : (float(latest.get("pm10",  0)), 150, "μg/m³", "#ef4444"),
    "NO₂"  : (float(latest.get("no2",   0)), 100, "μg/m³", "#8b5cf6"),
    "SO₂"  : (float(latest.get("so2",   0)), 75,  "μg/m³", "#06b6d4"),
    "O₃"   : (float(latest.get("o3",    0)), 100, "μg/m³", "#10b981"),
    "CO"   : (float(latest.get("co",    0)), 4,   "mg/m³", "#f97316"),
}

# ── 72-hour forecast via resolved model (Vertex AI or local) ──
with st.spinner("Generating 72-hour forecast…"):
    forecast_df = generate_forecast(raw_df, predict_fn, feature_cols, hours=72)

# Day summaries
def day_summary(df, day_offset):
    base  = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start = base + timedelta(days=day_offset)
    end   = start + timedelta(days=1)
    sub   = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
    if sub.empty:
        sub = df.iloc[day_offset*24:(day_offset+1)*24]
    if sub.empty:
        return None
    return {"mean": sub["aqi"].mean(), "min": sub["aqi"].min(), "max": sub["aqi"].max()}

day1 = day_summary(forecast_df, 1)
day2 = day_summary(forecast_df, 2)
day3 = day_summary(forecast_df, 3)

# ─────────────────────────────────────────────────────────────
# ── HEADER
# ─────────────────────────────────────────────────────────────
hcol1, hcol2 = st.columns([3, 1], gap="large")
with hcol1:
    st.markdown("""
    <p class="page-title">🌿 Multan Air Quality Predictor</p>
    <p class="page-sub">Real-time monitoring · 72-hour ML forecast · SHAP explainability</p>
    """, unsafe_allow_html=True)
with hcol2:
    # ── SOURCE BADGES — shows REAL source, never misleading text ──
    is_live = "BigQuery" in data_source or "Vertex" in model_source
    pill_class = "status-live" if is_live else "status-fallback"
    dot_class  = "dot-green"  if is_live else "dot-amber"

    # Data timestamp exactly from the data
    st.markdown(f"""
    <div style="text-align:right; padding-top:6px;">
        <div class="status-pill {pill_class}">
            <span class="dot {dot_class}"></span>{model_source}
        </div>
        <br/>
        <div class="source-badge">📊 Data: {data_source}</div>
        <p style="font-size:0.72rem; color:#334155; margin:6px 0 0;">
            Last data point: {last_updated}
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="hdivider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# ── ROW 1: AQI GAUGE  +  ALERT  +  POLLUTANTS
# ─────────────────────────────────────────────────────────────
col_gauge, col_alert, col_poll = st.columns([1.1, 1.5, 1.4], gap="large")

with col_gauge:
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_aqi,
        number=dict(font=dict(family="Manrope, sans-serif", size=52, color=current_info["color"])),
        gauge=dict(
            axis=dict(range=[0, 500], tickwidth=1, tickcolor="#334155",
                      tickfont=dict(color="#475569", size=9),
                      tickvals=[0,50,100,150,200,300,500]),
            bar=dict(color=current_info["color"], thickness=0.22),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[
                dict(range=[0,   50],  color="rgba(0,228,0,0.15)"),
                dict(range=[50,  100], color="rgba(212,184,0,0.15)"),
                dict(range=[100, 150], color="rgba(255,126,0,0.15)"),
                dict(range=[150, 200], color="rgba(255,0,0,0.15)"),
                dict(range=[200, 300], color="rgba(143,63,151,0.15)"),
                dict(range=[300, 500], color="rgba(126,0,35,0.15)"),
            ],
            threshold=dict(line=dict(color=current_info["color"], width=3),
                           thickness=0.78, value=current_aqi),
        ),
    ))
    gauge_fig.update_layout(**pl(height=220, margin=dict(l=16, r=16, t=24, b=0)))
    st.markdown('<div class="aqi-card">', unsafe_allow_html=True)
    st.markdown('<p class="metric-label">Current AQI · Multan, Pakistan</p>', unsafe_allow_html=True)
    st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(f"""
    <p style="text-align:center; margin:-10px 0 4px;">
        <span style="font-family:Manrope,sans-serif; font-size:1.1rem; font-weight:700;
                     color:{current_info['color']};">
            {current_info['icon']} {current_info['category']}
        </span>
    </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_alert:
    st.markdown('<div class="aqi-card" style="height:100%;">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Health Advisory</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="alert-box" style="background:{current_info['bg']};
         border-left-color:{current_info['color']}; margin-top:12px;">
        <span style="font-family:Manrope,sans-serif; font-size:1rem; font-weight:700;
                     color:{current_info['color']};">
            {current_info['icon']} {current_info['category'].upper()}
        </span>
        <p class="alert-msg">{current_info['message']}</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<p class="metric-label" style="margin-top:20px;">72h outlook</p>',
                unsafe_allow_html=True)
    avg_72 = forecast_df["aqi"].mean()
    mx_72  = forecast_df["aqi"].max()
    mn_72  = forecast_df["aqi"].min()
    snap_cols = st.columns(3)
    for col, label, val in zip(snap_cols, ["Avg AQI", "Peak", "Low"], [avg_72, mx_72, mn_72]):
        info = get_aqi_info(val)
        with col:
            st.markdown(f"""
            <div style="text-align:center; padding:10px 6px; background:#080d17;
                        border-radius:10px; border:1px solid rgba(255,255,255,0.05);">
                <p class="metric-label" style="margin:0 0 4px;">{label}</p>
                <span style="font-family:Manrope,sans-serif; font-size:1.6rem; font-weight:800;
                             color:{info['color']};">{int(val)}</span>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_poll:
    st.markdown('<div class="aqi-card" style="height:100%;">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Pollutant Levels</p>', unsafe_allow_html=True)
    st.markdown('<p class="metric-label" style="margin-bottom:14px;">Current concentrations vs WHO guidelines</p>',
                unsafe_allow_html=True)
    for name, (val, limit, unit, color) in pollutants.items():
        pct = min(val / limit * 100, 100)
        bar_color = color if pct < 80 else "#ef4444"
        st.markdown(f"""
        <div class="poll-row">
            <span class="poll-name">{name}</span>
            <div class="poll-bar-bg">
                <div class="poll-bar-fill" style="width:{pct:.0f}%;background:{bar_color};"></div>
            </div>
            <span class="poll-val">{val:.1f} {unit}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="hdivider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# ── ROW 2: 72-HOUR FORECAST CHART
# ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">72-Hour AQI Forecast</p>', unsafe_allow_html=True)
st.markdown("""
<div class="legend-row">
  <div class="legend-item"><div class="legend-dot" style="background:#00E400;"></div>Good (0–50)</div>
  <div class="legend-item"><div class="legend-dot" style="background:#D4B800;"></div>Moderate (51–100)</div>
  <div class="legend-item"><div class="legend-dot" style="background:#FF7E00;"></div>Sensitive (101–150)</div>
  <div class="legend-item"><div class="legend-dot" style="background:#FF0000;"></div>Unhealthy (151–200)</div>
  <div class="legend-item"><div class="legend-dot" style="background:#8F3F97;"></div>Very Unhealthy (201–300)</div>
  <div class="legend-item"><div class="legend-dot" style="background:#7E0023;"></div>Hazardous (301+)</div>
</div>
""", unsafe_allow_html=True)

fdf = forecast_df.copy()
forecast_fig = go.Figure()
for i in range(len(fdf) - 1):
    segment = fdf.iloc[i:i+2]
    seg_col  = get_aqi_info(segment["aqi"].mean())["color"]
    forecast_fig.add_trace(go.Scatter(
        x=segment["timestamp"], y=segment["aqi"],
        mode="lines", line=dict(color=seg_col, width=2.5),
        showlegend=False, hoverinfo="skip",
    ))
forecast_fig.add_trace(go.Scatter(
    x=fdf["timestamp"], y=fdf["aqi"],
    mode="markers",
    marker=dict(size=5, color=fdf["aqi"].apply(lambda x: get_aqi_info(x)["color"]),
                line=dict(width=1, color="#0a0f1e")),
    showlegend=False,
    hovertemplate="<b>%{x|%a %d %b %H:%M}</b><br>AQI: <b>%{y:.0f}</b><extra></extra>",
))
for day in range(1, 4):
    day_ts = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=day)
    forecast_fig.add_vline(x=day_ts, line_dash="dot", line_color="rgba(255,255,255,0.12)",
                           line_width=1, annotation_text=f"Day {day}",
                           annotation_font=dict(color="#475569", size=10),
                           annotation_position="top right")
forecast_fig.update_layout(**pl(
    height=280, margin=dict(l=12, r=12, t=36, b=12),
    shapes=AQI_PLOTLY_ZONES,
    yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="AQI", range=[0, max(fdf["aqi"].max()*1.1, 200)]),
    xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title=""),
    hovermode="x unified",
))
st.plotly_chart(forecast_fig, use_container_width=True, config={"displayModeBar": False})

# ─────────────────────────────────────────────────────────────
# ── ROW 3: DAY SUMMARY CARDS
# ─────────────────────────────────────────────────────────────
day_labels = [
    (datetime.now() + timedelta(days=1)).strftime("Tomorrow · %a %d %b"),
    (datetime.now() + timedelta(days=2)).strftime("Day 2 · %a %d %b"),
    (datetime.now() + timedelta(days=3)).strftime("Day 3 · %a %d %b"),
]
dcols = st.columns(3, gap="medium")
for col, label, summary in zip(dcols, day_labels, [day1, day2, day3]):
    with col:
        if summary is None:
            continue
        info = get_aqi_info(summary["mean"])
        st.markdown(f"""
        <div class="day-card">
            <p class="day-label">{label}</p>
            <p class="day-aqi" style="color:{info['color']};">{int(summary['mean'])}</p>
            <span class="day-cat" style="background:{info['bg']};color:{info['color']};">
                {info['icon']} {info['category']}
            </span>
            <div class="range-row">
                <div class="range-item">↓ <span class="range-val">{int(summary['min'])}</span></div>
                <div class="range-item">↑ <span class="range-val">{int(summary['max'])}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<hr class="hdivider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# ── ROW 4: HISTORICAL TREND  +  HOURLY HEATMAP
# ─────────────────────────────────────────────────────────────
hist_col, heat_col = st.columns([1.6, 1], gap="large")

with hist_col:
    st.markdown('<p class="section-title">Historical Trend — Last 7 Days</p>', unsafe_allow_html=True)
    hist = raw_df.tail(168).copy()
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Scatter(
        x=hist["timestamp"], y=hist["aqi"],
        mode="lines", line=dict(color="#38bdf8", width=1.5),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.06)",
        name="Observed AQI",
        hovertemplate="<b>%{x|%d %b %H:%M}</b><br>AQI: <b>%{y}</b><extra></extra>",
    ))
    hist_fig.update_layout(**pl(
        height=230, margin=dict(l=12, r=12, t=12, b=12),
        shapes=AQI_PLOTLY_ZONES,
        yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="AQI"),
        xaxis=dict(**PLOTLY_LAYOUT["xaxis"]),
    ))
    st.plotly_chart(hist_fig, use_container_width=True, config={"displayModeBar": False})

with heat_col:
    st.markdown('<p class="section-title">Hourly AQI Pattern</p>', unsafe_allow_html=True)
    hourly = raw_df.tail(24*30).copy()
    hourly["hour"] = pd.to_datetime(hourly["timestamp"]).dt.hour
    hourly["dow"]  = pd.to_datetime(hourly["timestamp"]).dt.day_name().str[:3]
    pivot = hourly.groupby(["dow","hour"])["aqi"].mean().unstack(fill_value=0)
    day_order = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    pivot = pivot.reindex([d for d in day_order if d in pivot.index])
    heat_fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=[
            [0.0, "#00E400"],[0.1, "#D4B800"],[0.3, "#FF7E00"],
            [0.5, "#FF0000"],[0.75,"#8F3F97"],[1.0, "#7E0023"]
        ],
        zmin=0, zmax=300, showscale=True,
        hovertemplate="<b>%{y} %{x}</b><br>Avg AQI: <b>%{z:.0f}</b><extra></extra>",
        colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(color="#475569", size=9)),
    ))
    heat_fig.update_layout(**pl(
        height=230, margin=dict(l=12, r=12, t=12, b=12),
        xaxis=dict(**PLOTLY_LAYOUT["xaxis"],
                   tickvals=list(range(0,24,4)),
                   ticktext=[f"{h:02d}:00" for h in range(0,24,4)]),
    ))
    st.plotly_chart(heat_fig, use_container_width=True, config={"displayModeBar": False})

st.markdown('<hr class="hdivider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# ── ROW 5: SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Model Explainability — SHAP Feature Importance</p>',
            unsafe_allow_html=True)
st.markdown('<p style="font-size:0.82rem; color:#475569; margin:-2px 0 16px;">'
            'SHAP values show each feature\'s contribution to the AQI prediction. '
            'Red = increases AQI · Blue = decreases AQI.</p>', unsafe_allow_html=True)

shap_path, shap_vals, top_feats = compute_shap(local_model, local_scaler, feature_cols, raw_df)
shap_img_col, shap_tbl_col = st.columns([1.7, 1], gap="large")

with shap_img_col:
    if shap_path and Path(shap_path).exists():
        st.image(shap_path, use_container_width=True)
    else:
        st.info("SHAP plot unavailable — run explainability.py to generate it.")

with shap_tbl_col:
    st.markdown('<p class="section-title" style="font-size:0.9rem;">Top Contributing Features</p>',
                unsafe_allow_html=True)
    if top_feats:
        max_val = top_feats[0][1] if top_feats else 1
        for rank, (fname, importance) in enumerate(top_feats, 1):
            pct = importance / max_val * 100
            clean = fname.replace("_", " ").title()
            bar_col = "#38bdf8" if rank <= 3 else "#1e40af"
            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                    <span style="font-size:0.8rem; color:#94a3b8;">{rank}. {clean}</span>
                    <span style="font-size:0.78rem; color:#475569;">{importance:.3f}</span>
                </div>
                <div style="background:#1e293b; border-radius:3px; height:6px;">
                    <div style="width:{pct:.0f}%; height:100%; background:{bar_col};
                                border-radius:3px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <p style="font-size:0.82rem; color:#475569;">
        Top SHAP features for AQI prediction typically include:<br><br>
        • <b>aqi_lag_1h</b> — Previous hour AQI<br>
        • <b>aqi_roll_mean_24h</b> — 24h rolling average<br>
        • <b>pm2_5</b> — Fine particulate matter<br>
        • <b>aqi_lag_24h</b> — Same hour, yesterday<br>
        • <b>hour_sin / hour_cos</b> — Time of day pattern
        </p>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# ── FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown('<hr class="hdivider">', unsafe_allow_html=True)
st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center; padding: 4px 0 12px;">
    <span style="font-size:0.75rem; color:#334155;">
        Pearls AQI Predictor · Multan, Pakistan ·
        XGBoost R²=0.989 · Inference: {model_source} · Data: {data_source}
    </span>
    <span style="font-size:0.75rem; color:#334155;">
        AQI scale follows EPA standard · 72-hour recursive ML forecast
    </span>
</div>
""", unsafe_allow_html=True)
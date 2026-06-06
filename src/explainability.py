"""
Explains an AQI prediction model using SHAP values.
Reads data from BigQuery, engineers features, and generates a summary plot.
"""

import json, pickle, warnings
from pathlib import Path

import joblib, shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from google.cloud import bigquery, storage


# ── CONFIGURATION ─────────────────────────────────────────────────────────────
PROJECT_ID  = "pearl-aqi-predictor"
DATASET_ID  = "aqi_feature_store"
TABLE_ID    = "multan_historical"
GCS_BUCKET  = "pearl-aqi-predictor-artifacts"
GCS_PREFIX  = "artifacts"

ROOT        = Path(__file__).resolve().parent.parent
ARTIFACTS   = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


# ── 1. LOAD RAW DATA FROM BIGQUERY ────────────────────────────────────────────
print("Loading data from BigQuery...")
bq_client = bigquery.Client(project=PROJECT_ID)
query = f"""
    SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
    ORDER BY timestamp ASC
"""
df = bq_client.query(query).to_dataframe()
print(f"Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")


# ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────────
print("Engineering features...")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Lag features
for lag in [1, 2, 3, 6, 12, 24, 48, 72]:
    df[f"aqi_lag_{lag}h"] = df["aqi"].shift(lag)

for lag in [1, 3, 6, 24]:
    df[f"pm25_lag_{lag}h"] = df["pm2_5"].shift(lag)
    df[f"co_lag_{lag}h"]   = df["co"].shift(lag)

# Rolling statistics
for window in [3, 6, 12, 24]:
    df[f"aqi_roll_mean_{window}h"] = df["aqi"].rolling(window).mean()
    df[f"aqi_roll_std_{window}h"]  = df["aqi"].rolling(window).std()

for window in [6, 24]:
    df[f"pm2_5_roll_mean_{window}h"] = df["pm2_5"].rolling(window).mean()
    df[f"co_roll_mean_{window}h"]    = df["co"].rolling(window).mean()
    df[f"no2_roll_mean_{window}h"]   = df["no2"].rolling(window).mean()

# Diff features
for diff in [1, 3, 24]:
    df[f"aqi_diff_{diff}h"] = df["aqi"].diff(diff)

# Trend
df["aqi_trend"] = df["aqi"].diff(1).rolling(3).mean()

# Time features
df["hour"]        = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"]       = df["timestamp"].dt.month
df["day_of_year"] = df["timestamp"].dt.dayofyear

# Cyclical encoding
df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

# Peak hours
df["is_morning_peak"] = df["hour"].between(7, 9).astype(int)
df["is_evening_peak"] = df["hour"].between(17, 20).astype(int)

# Season
df["season"] = df["month"].map(
    {12: 0, 1: 0, 2: 0,
      3: 1, 4: 1, 5: 1,
      6: 2, 7: 2, 8: 2,
      9: 3, 10: 3, 11: 3}
)

# Drop NaN rows from lag/rolling, keep last 500 for SHAP efficiency
df = df.dropna().tail(500).reset_index(drop=True)
print(f"After feature engineering: {df.shape[0]} rows, {df.shape[1]} columns")


# ── 3. LOAD ARTIFACTS FROM GCS ────────────────────────────────────────────────
print("Downloading artifacts from Google Cloud Storage...")
gcs_client = storage.Client(project=PROJECT_ID)
bucket     = gcs_client.bucket(GCS_BUCKET)

def download_artifact(filename):
    blob = bucket.blob(f"{GCS_PREFIX}/{filename}")
    local_path = ARTIFACTS / filename
    blob.download_to_filename(local_path)
    print(f"  Downloaded: {filename}")
    return local_path


# ── 4. LOAD FEATURE NAMES ─────────────────────────────────────────────────────
feature_path = download_artifact("feature_cols.json")
with open(feature_path) as f:
    raw = json.load(f)
cols = raw if isinstance(raw, list) else raw["features"]
print(f"Features: {len(cols)} columns")


# ── 5. LOAD SCALER ────────────────────────────────────────────────────────────
scaler_path = download_artifact("aqi_scaler.joblib")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    scaler = joblib.load(scaler_path)
print("Scaler loaded.")


# ── 6. LOAD MODEL ─────────────────────────────────────────────────────────────
model_path = download_artifact("aqi_best_model.pkl")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    model = pickle.load(open(model_path, "rb"))
print("Model loaded.")


# ── 7. SCALE FEATURES ─────────────────────────────────────────────────────────
X = pd.DataFrame(scaler.transform(df[cols]), columns=cols)


# ── 8. COMPUTE SHAP VALUES ────────────────────────────────────────────────────
print("Computing SHAP values...")
explainer   = shap.Explainer(model, X)
shap_values = explainer(X)
print("SHAP values computed.")


# ── 9. PLOT & UPLOAD TO GCS ───────────────────────────────────────────────────
plot_path = ARTIFACTS / "shap_summary.png"
shap.summary_plot(shap_values, X, show=False)
plt.savefig(plot_path, bbox_inches="tight", dpi=150)
plt.close()
print(f"Plot saved locally → {plot_path}")

blob = bucket.blob(f"{GCS_PREFIX}/shap_summary.png")
blob.upload_from_filename(plot_path)
print(f"Plot uploaded to GCS → gs://{GCS_BUCKET}/{GCS_PREFIX}/shap_summary.png")
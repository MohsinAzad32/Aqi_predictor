"""
explainability.py
-----------------
Explains an AQI prediction model using SHAP values.
Reads data from BigQuery and artifacts from Google Cloud Storage.
Generates a summary plot showing which features matter most.
"""

import json, pickle, warnings, tempfile
from pathlib import Path

import joblib, shap
import matplotlib
matplotlib.use("Agg")          # ← use file-based backend, no GUI/Tkinter needed
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from google.cloud import bigquery, storage


# ── CONFIGURATION ─────────────────────────────────────────────────────────────
PROJECT_ID  = "pearl-aqi-predictor"
DATASET_ID  = "aqi_feature_store"
TABLE_ID    = "multan_historical"
GCS_BUCKET  = "pearl-aqi-predictor-artifacts"   # ← change to your GCS bucket name
GCS_PREFIX  = "artifacts"                        # ← folder inside the bucket

ROOT        = Path(__file__).resolve().parent.parent
ARTIFACTS   = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


# ── 1. LOAD DATA FROM BIGQUERY ────────────────────────────────────────────────
print("Loading data from BigQuery...")
bq_client = bigquery.Client(project=PROJECT_ID)
query = f"""
    SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
    ORDER BY timestamp DESC
"""
df = bq_client.query(query).to_dataframe()
print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")


# ── 2. LOAD ARTIFACTS FROM GCS ────────────────────────────────────────────────
print("Downloading artifacts from Google Cloud Storage...")
gcs_client = storage.Client(project=PROJECT_ID)
bucket     = gcs_client.bucket(GCS_BUCKET)

def download_artifact(filename):
    """Download a file from GCS to local artifacts folder."""
    blob = bucket.blob(f"{GCS_PREFIX}/{filename}")
    local_path = ARTIFACTS / filename
    blob.download_to_filename(local_path)
    print(f"  Downloaded: {filename}")
    return local_path


# ── 3. LOAD FEATURE NAMES ─────────────────────────────────────────────────────
feature_path = download_artifact("feature_cols.json")
with open(feature_path) as f:
    raw = json.load(f)
cols = raw if isinstance(raw, list) else raw["features"]
print(f"Features: {len(cols)} columns")


# ── 4. LOAD SCALER ────────────────────────────────────────────────────────────
scaler_path = download_artifact("aqi_scaler.joblib")
scaler = joblib.load(scaler_path)
print("Scaler loaded.")


# ── 5. LOAD MODEL ─────────────────────────────────────────────────────────────
model_path = download_artifact("aqi_best_model.pkl")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    model = pickle.load(open(model_path, "rb"))
print("Model loaded.")


# ── 6. SCALE FEATURES ─────────────────────────────────────────────────────────
X = pd.DataFrame(scaler.transform(df[cols]), columns=cols)


# ── 7. COMPUTE SHAP VALUES ────────────────────────────────────────────────────
explainer   = shap.Explainer(model, X)
shap_values = explainer(X)
print("SHAP values computed.")


# ── 8. PLOT & SAVE TO GCS ─────────────────────────────────────────────────────
plot_path = ARTIFACTS / "shap_summary.png"
shap.summary_plot(shap_values, X, show=False)
plt.savefig(plot_path, bbox_inches="tight", dpi=150)
plt.close()
print(f"Plot saved locally → {plot_path}")

# Upload plot back to GCS
blob = bucket.blob(f"{GCS_PREFIX}/shap_summary.png")
blob.upload_from_filename(plot_path)
print(f"Plot uploaded to GCS → gs://{GCS_BUCKET}/{GCS_PREFIX}/shap_summary.png")
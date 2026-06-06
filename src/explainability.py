"""
explainability.py
-----------------
Explains an AQI prediction model using SHAP values.
Generates a summary plot showing which features matter most.
"""

import json, pickle, warnings
from pathlib import Path

import joblib, shap
import matplotlib
matplotlib.use("Agg")          # ← use file-based backend, no GUI/Tkinter needed
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb


# ── 1. PATHS 
# Build absolute paths from this file's location so the script works
# whether you run it from src/ or the project root.
ROOT      = Path(__file__).resolve().parent.parent   # E:\AQI_Predictor
ARTIFACTS = ROOT / "artifacts"
DATA      = ROOT / "data" / "multan_features.csv"


# ── 2. LOAD DATA 
df = pd.read_csv(DATA)
print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")


# ── 3. LOAD FEATURE NAMES 
# JSON can be a plain list  ["col1", "col2"]
# or a dict                 {"features": ["col1", "col2"]}
with open(ARTIFACTS / "feature_cols.json") as f:
    raw = json.load(f)
cols = raw if isinstance(raw, list) else raw["features"]
print(f"Features: {len(cols)} columns")


# ── 4. LOAD SCALER 
# joblib is the sklearn-recommended way to save/load scalers.
scaler = joblib.load(ARTIFACTS / "aqi_scaler.joblib")
print("Scaler loaded.")


# ── 5. LOAD MODEL 
# Suppress the version-mismatch warning from older pickle-saved XGBoost models.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    model = pickle.load(open(ARTIFACTS / "aqi_best_model.pkl", "rb"))
print("Model loaded.")


# ── 6. SCALE FEATURES ─────────────────────────────────────────────────────────
X = pd.DataFrame(scaler.transform(df[cols]), columns=cols)


# ── 7. COMPUTE SHAP VALUES ────────────────────────────────────────────────────
# SHAP explains *why* the model made each prediction.
# Each feature gets a SHAP value: positive = pushed prediction up,
#                                  negative = pushed prediction down.
explainer   = shap.Explainer(model, X)
shap_values = explainer(X)
print("SHAP values computed.")


# ── 8. PLOT & SAVE ────────────────────────────────────────────────────────────
# The summary plot ranks features by importance and shows their effect direction.
shap.summary_plot(shap_values, X, show=False)
plt.savefig(ARTIFACTS / "shap_summary.png", bbox_inches="tight", dpi=150)
plt.close()
print(f"Plot saved → {ARTIFACTS / 'shap_summary.png'}")
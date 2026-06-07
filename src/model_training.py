# ============================================
# model_training.py
# Retrain AQI model on correct EPA AQI data
# Run once locally: python src/model_training.py
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, json, pickle, warnings
warnings.filterwarnings('ignore')

from google.cloud import bigquery, storage
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
PROJECT_ID  = "pearl-aqi-predictor"
GCS_BUCKET  = "pearl-aqi-predictor-artifacts"
GCS_PREFIX  = "artifacts"
ARTIFACTS   = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# CELL 2: Load Data from BigQuery (uses ADC — no browser popup)
# ─────────────────────────────────────────────────────────────
print("Connecting to BigQuery via Application Default Credentials...")
bq_client = bigquery.Client(project=PROJECT_ID)

print("Downloading Multan AQI data from BigQuery...")
query = f"""
    SELECT *
    FROM `{PROJECT_ID}.aqi_feature_store.multan_historical`
    ORDER BY timestamp ASC
"""
df = bq_client.query(query).to_dataframe()
print(f"\nData loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"AQI range: {df['aqi'].min():.0f} → {df['aqi'].max():.0f}  (mean: {df['aqi'].mean():.1f})")
print(df.head())

# ─────────────────────────────────────────────────────────────
# CELL 3-4: Clean and Prepare
# ─────────────────────────────────────────────────────────────
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
df = df.set_index("timestamp")

print(f"\nDate range: {df.index.min()} → {df.index.max()}")
print(f"Total hours: {len(df):,}")

# ─────────────────────────────────────────────────────────────
# CELL 6: Interpolate Missing Values
# ─────────────────────────────────────────────────────────────
df_clean = df.copy()
pollutants = ['pm2_5', 'pm10', 'no2', 'so2', 'o3', 'co', 'aqi']

for col in pollutants:
    n_missing = df_clean[col].isnull().sum()
    if n_missing > 0:
        print(f"  {col}: {n_missing} missing → interpolating")
        df_clean[col] = df_clean[col].interpolate(method='time', limit=3)

before = len(df_clean)
df_clean = df_clean.dropna(subset=pollutants)
print(f"Dropped {before - len(df_clean)} rows with large gaps")

# ─────────────────────────────────────────────────────────────
# CELL 7: Cap Outliers
# ─────────────────────────────────────────────────────────────
def cap_outliers_iqr(series, factor=3.0):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return series.clip(lower=Q1 - factor*IQR, upper=Q3 + factor*IQR)

for col in pollutants:
    df_clean[col] = cap_outliers_iqr(df_clean[col])

# ─────────────────────────────────────────────────────────────
# CELL 11: Clip Negatives
# ─────────────────────────────────────────────────────────────
for col in ['pm2_5', 'pm10', 'no2', 'so2', 'o3', 'co']:
    df_clean[col] = df_clean[col].clip(lower=0)

print(f"\nAfter cleaning: {len(df_clean):,} rows")
print(f"AQI range: {df_clean['aqi'].min():.0f} → {df_clean['aqi'].max():.0f}")

# ─────────────────────────────────────────────────────────────
# CELL 15-19: Feature Engineering
# ─────────────────────────────────────────────────────────────
print("\nEngineering features...")
df_feat = df_clean.copy()

if 'city' in df_feat.columns:
    df_feat = df_feat.drop(columns=['city'])

# Lag features
for lag in [1, 2, 3, 6, 12, 24, 48, 72]:
    df_feat[f'aqi_lag_{lag}h'] = df_feat['aqi'].shift(lag)

for lag in [1, 3, 6, 24]:
    df_feat[f'pm25_lag_{lag}h'] = df_feat['pm2_5'].shift(lag)
    df_feat[f'co_lag_{lag}h']   = df_feat['co'].shift(lag)

# Rolling features
for window in [3, 6, 12, 24]:
    df_feat[f'aqi_roll_mean_{window}h'] = df_feat['aqi'].shift(1).rolling(window).mean()
    df_feat[f'aqi_roll_std_{window}h']  = df_feat['aqi'].shift(1).rolling(window).std()

for col in ['pm2_5', 'co', 'no2']:
    for window in [6, 24]:
        df_feat[f'{col}_roll_mean_{window}h'] = df_feat[col].shift(1).rolling(window).mean()

# Diff features
df_feat['aqi_diff_1h']  = df_feat['aqi'].shift(1).diff(1)
df_feat['aqi_diff_3h']  = df_feat['aqi'].shift(1).diff(3)
df_feat['aqi_diff_24h'] = df_feat['aqi'].shift(1).diff(24)
df_feat['aqi_trend']    = np.sign(df_feat['aqi_diff_1h'])

# Time features
df_feat['hour']        = df_feat.index.hour
df_feat['day_of_week'] = df_feat.index.dayofweek
df_feat['month']       = df_feat.index.month
df_feat['day_of_year'] = df_feat.index.dayofyear

df_feat['hour_sin']  = np.sin(2 * np.pi * df_feat['hour'] / 24)
df_feat['hour_cos']  = np.cos(2 * np.pi * df_feat['hour'] / 24)
df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
df_feat['dow_sin']   = np.sin(2 * np.pi * df_feat['day_of_week'] / 7)
df_feat['dow_cos']   = np.cos(2 * np.pi * df_feat['day_of_week'] / 7)

df_feat['is_morning_peak'] = df_feat['hour'].between(6, 9).astype(int)
df_feat['is_evening_peak'] = df_feat['hour'].between(19, 22).astype(int)

def get_season(month):
    if month in [12, 1, 2]:   return 0
    if month in [3, 4, 5]:    return 1
    if month in [6, 7, 8, 9]: return 2
    return 3

df_feat['season'] = df_feat['month'].apply(get_season)

# Drop NaN
before = len(df_feat)
df_feat = df_feat.dropna()
print(f"Rows after feature engineering: {len(df_feat):,} (dropped {before-len(df_feat):,} NaN rows)")

# ─────────────────────────────────────────────────────────────
# CELL 24-25: Train/Test Split and Scaling
# ─────────────────────────────────────────────────────────────
TARGET       = 'aqi'
FEATURE_COLS = [c for c in df_feat.columns if c != TARGET]

X = df_feat[FEATURE_COLS]
y = df_feat[TARGET]

split_idx = int(len(df_feat) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\nTrain: {len(X_train):,} rows  ({X_train.index.min().date()} → {X_train.index.max().date()})")
print(f"Test : {len(X_test):,}  rows  ({X_test.index.min().date()} → {X_test.index.max().date()})")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────
# CELL 31: Train XGBoost (Best Model)
# ─────────────────────────────────────────────────────────────
print("\nTraining XGBoost model...")

xgb_model = XGBRegressor(
    n_estimators          = 1000,
    learning_rate         = 0.05,
    max_depth             = 6,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    min_child_weight      = 5,
    reg_alpha             = 0.1,
    reg_lambda            = 1.0,
    random_state          = 42,
    n_jobs                = -1,
    early_stopping_rounds = 50,
    eval_metric           = 'mae',
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100,
)

preds = np.clip(xgb_model.predict(X_test), 0, 500)
mae   = mean_absolute_error(y_test, preds)
rmse  = np.sqrt(mean_squared_error(y_test, preds))
r2    = r2_score(y_test, preds)

print(f"\n{'='*45}")
print(f"  XGBoost Results on Correct EPA AQI Data")
print(f"{'='*45}")
print(f"  MAE  : {mae:.2f}")
print(f"  RMSE : {rmse:.2f}")
print(f"  R²   : {r2:.4f}")
print(f"  Best iteration: {xgb_model.best_iteration}")
print(f"{'='*45}")

# ─────────────────────────────────────────────────────────────
# CELL 34: Save Artifacts Correctly
# ─────────────────────────────────────────────────────────────
print("\nSaving artifacts...")

# Save model with pickle (correct — saves the model OBJECT not the module)
model_path = ARTIFACTS / "aqi_best_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(xgb_model, f)
print(f"  ✅ Model saved: {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")

# Save scaler with joblib
scaler_path = ARTIFACTS / "aqi_scaler.joblib"
joblib.dump(scaler, scaler_path)
print(f"  ✅ Scaler saved: {scaler_path}")

# Save feature columns
feature_path = ARTIFACTS / "feature_cols.json"
with open(feature_path, "w") as f:
    json.dump(FEATURE_COLS, f)
print(f"  ✅ Feature cols saved: {feature_path} ({len(FEATURE_COLS)} features)")

# ─────────────────────────────────────────────────────────────
# Upload to GCS
# ─────────────────────────────────────────────────────────────
print(f"\nUploading artifacts to GCS bucket: {GCS_BUCKET}...")
gcs_client = storage.Client(project=PROJECT_ID)
bucket     = gcs_client.bucket(GCS_BUCKET)

for fname in ["aqi_best_model.pkl", "aqi_scaler.joblib", "feature_cols.json"]:
    blob = bucket.blob(f"{GCS_PREFIX}/{fname}")
    blob.upload_from_filename(ARTIFACTS / fname)
    print(f"  ✅ Uploaded: gs://{GCS_BUCKET}/{GCS_PREFIX}/{fname}")

print(f"\nRetraining complete!")
print(f"   Model R²  : {r2:.4f}")
print(f"   Model MAE : {mae:.2f} AQI points")
print(f"   Features  : {len(FEATURE_COLS)}")
print(f"   Training data: {len(X_train):,} rows of correct EPA AQI")
print(f"\nRestart Streamlit to use the new model!")
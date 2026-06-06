# src/tests/test_data.py
import pytest
import pandas as pd
import numpy as np


def test_no_negative_pollutants():
    """Pollutant concentrations must never be negative."""
    from google.cloud import bigquery
    import os

    # Skip in CI if no credentials available
    if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS') and \
       not os.getenv('GCP_SA_KEY'):
        pytest.skip("No GCP credentials in environment")

    # Use a small sample to keep the test fast
    client  = bigquery.Client(project="pearl-aqi-predictor")
    query   = """
        SELECT pm2_5, pm10, no2, so2, o3, co
        FROM `pearl-aqi-predictor.aqi_feature_store.multan_features`
        LIMIT 1000
    """
    df = client.query(query).to_dataframe()

    for col in ['pm2_5', 'pm10', 'no2', 'so2', 'o3', 'co']:
        assert (df[col] >= 0).all(), f"{col} has negative values"


def test_aqi_range():
    """Recomputed AQI must be within 0–500."""
    # Test with synthetic data using the AQI formula
    from src.predict import forecast_72h
    assert True   # placeholder — extend with real AQI formula test


def test_feature_cols_match_model():
    """Feature columns file must match model's expected input."""
    import joblib, json, os

    model_path    = os.getenv('MODEL_PATH', 'src/models/aqi_best_model.pkl')
    features_path = os.getenv('FEATURES_PATH', 'src/models/feature_cols.json')

    model = joblib.load(model_path)
    with open(features_path) as f:
        feature_cols = json.load(f)

    # XGBoost stores n_features_in_
    assert model.n_features_in_ == len(feature_cols), \
        f"Model expects {model.n_features_in_} features, got {len(feature_cols)}"
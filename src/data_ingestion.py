import os
import pandas as pd
import requests
import datetime
from google.cloud import bigquery
from google.cloud import aiplatform
import pydata_google_auth
from dotenv import load_dotenv

# 1. Load API Keys and Environment Variables
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not API_KEY:
    raise ValueError("API Key not Found. Check your .env file")

# Coordinates for Multan
LAT = 30.1575
LON = 71.5249

# --- PRODUCTION FIX 1: Use CURRENT API instead of Bulk History ---
# Pulls only the immediate, most recent hourly measurement point
URL = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"

print("Fetching Fresh Hourly Real-Time Data from OpenWeather...")
response = requests.get(URL)

if response.status_code != 200:
    raise Exception(f"API Failed with status code {response.status_code}: {response.text}")

data = response.json()

# 2. Normalize and Clean the Single-Row Feature Vector
df = pd.json_normalize(data["list"])
df.columns = df.columns.str.replace("components.", "", regex=False)
df.columns = df.columns.str.replace("main.", "", regex=False)

df["city"] = "Multan"

# --- PRODUCTION FIX 2: Explicit Native Timestamp Handling ---
# Ensures your schema views align with a standard datetime string representation
df["timestamp"] = pd.to_datetime(datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))

core_columns = ["city", "timestamp", "aqi", "pm2_5", "pm10", "no2", "so2", "o3", "co"]
available_columns = [col for col in core_columns if col in df.columns]
df = df[available_columns]

print("New Feature Vector Extracted:")
print(df)

# ---------------------------------------------------------
# 3. Secure Cloud Infrastructure Data Stream Routing
# ---------------------------------------------------------
PROJECT_ID = "pearl-aqi-predictor"
REGION = "us-central1"
DATASET_ID = "aqi_feature_store"
TABLE_ID = "multan_historical"
table_path = f"{DATASET_ID}.{TABLE_ID}"

print("\n🔒 Connecting to Google Cloud Platform...")

# Kept your brilliant web browser fallback auth logic intact for seamless access!
credentials = pydata_google_auth.get_user_credentials(
    ['https://www.googleapis.com/auth/cloud-platform']
)

aiplatform.init(project=PROJECT_ID, location=REGION, credentials=credentials)
bq_client = bigquery.Client(project=PROJECT_ID, credentials=credentials)

# --- PRODUCTION FIX 3: Append Mode Instead of Nuke & Replace ---
print(f"Streaming new telemetry vector directly to GCP table: {table_path}...")
df.to_gbq(
    destination_table=table_path, 
    project_id=PROJECT_ID, 
    credentials=credentials,
    if_exists="append"  # Changed from 'replace' to safely stack new hours chronological data!
)

print(f"Success! Ingestion task finished. Row appended to BigQuery table layer.")
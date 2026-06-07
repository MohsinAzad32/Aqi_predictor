import os
import pandas as pd
import requests
import datetime
from google.cloud import bigquery
from google.cloud import aiplatform
from dotenv import load_dotenv
import pandas_gbq

# ─────────────────────────────────────────────────────────────
# 1. Load API Keys and Environment Variables
# ─────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not API_KEY:
    raise ValueError("API Key not Found. Check your .env file or GitHub Secrets")

# ─────────────────────────────────────────────────────────────
# 2. EPA AQI Calculator
# OpenWeather units:
#   pm2_5, pm10, no2, so2, o3 → μg/m³
#   co                        → μg/m³  ← NOT mg/m³ !
# ─────────────────────────────────────────────────────────────
def calculate_epa_aqi(pm25=None, pm10=None, no2=None, o3=None, so2=None, co=None):
    """
    Calculate EPA AQI from pollutant concentrations.
    Returns the highest sub-index across all pollutants (EPA standard).
    """
    def linear(Cp, bp_lo, bp_hi, aqi_lo, aqi_hi):
        return round(((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (Cp - bp_lo) + aqi_lo)

    def aqi_pm25(c):
        """PM2.5 in μg/m³"""
        bp = [
            (0.0,   12.0,   0,  50),
            (12.1,  35.4,  51, 100),
            (35.5,  55.4, 101, 150),
            (55.5, 150.4, 151, 200),
            (150.5,250.4, 201, 300),
            (250.5,350.4, 301, 400),
            (350.5,500.4, 401, 500),
        ]
        for lo, hi, alo, ahi in bp:
            if lo <= c <= hi:
                return linear(c, lo, hi, alo, ahi)
        return 500

    def aqi_pm10(c):
        """PM10 in μg/m³"""
        bp = [
            (0,   54,   0,  50),
            (55,  154,  51, 100),
            (155, 254, 101, 150),
            (255, 354, 151, 200),
            (355, 424, 201, 300),
            (425, 504, 301, 400),
            (505, 604, 401, 500),
        ]
        for lo, hi, alo, ahi in bp:
            if lo <= c <= hi:
                return linear(c, lo, hi, alo, ahi)
        return 500

    def aqi_no2(c):
        """NO2 in μg/m³ → ppb (divide by 1.88)"""
        c = c / 1.88
        bp = [
            (0,    53,   0,  50),
            (54,  100,  51, 100),
            (101, 360, 101, 150),
            (361, 649, 151, 200),
            (650,1249, 201, 300),
            (1250,1649,301, 400),
            (1650,2049,401, 500),
        ]
        for lo, hi, alo, ahi in bp:
            if lo <= c <= hi:
                return linear(c, lo, hi, alo, ahi)
        return 500

    def aqi_o3(c):
        """O3 in μg/m³ → ppb (divide by 1.96)"""
        c = c / 1.96
        bp = [
            (0,   54,   0,  50),
            (55,  70,  51, 100),
            (71,  85, 101, 150),
            (86, 105, 151, 200),
            (106,200, 201, 300),
        ]
        for lo, hi, alo, ahi in bp:
            if lo <= c <= hi:
                return linear(c, lo, hi, alo, ahi)
        return 300

    def aqi_so2(c):
        """SO2 in μg/m³ → ppb (divide by 2.62)"""
        c = c / 2.62
        bp = [
            (0,   35,   0,  50),
            (36,  75,  51, 100),
            (76, 185, 101, 150),
            (186,304, 151, 200),
            (305,604, 201, 300),
            (605,804, 301, 400),
            (805,1004,401, 500),
        ]
        for lo, hi, alo, ahi in bp:
            if lo <= c <= hi:
                return linear(c, lo, hi, alo, ahi)
        return 500

    def aqi_co(c):
        """
        CO from OpenWeather is in μg/m³
        Step 1: μg/m³ → mg/m³ (divide by 1000)
        Step 2: mg/m³ → ppm  (divide by 1.145)
        """
        c = c / 1000    # ← KEY FIX: μg/m³ to mg/m³
        c = c / 1.145   # mg/m³ to ppm
        bp = [
            (0,    4.4,   0,  50),
            (4.5,  9.4,  51, 100),
            (9.5,  12.4,101, 150),
            (12.5, 15.4,151, 200),
            (15.5, 30.4,201, 300),
            (30.5, 40.4,301, 400),
            (40.5, 50.4,401, 500),
        ]
        for lo, hi, alo, ahi in bp:
            if lo <= c <= hi:
                return linear(c, lo, hi, alo, ahi)
        return 500

    scores = []
    if pm25 is not None: scores.append(aqi_pm25(pm25))
    if pm10  is not None: scores.append(aqi_pm10(pm10))
    if no2   is not None: scores.append(aqi_no2(no2))
    if o3    is not None: scores.append(aqi_o3(o3))
    if so2   is not None: scores.append(aqi_so2(so2))
    if co    is not None: scores.append(aqi_co(co))

    return max(scores) if scores else 0


# ─────────────────────────────────────────────────────────────
# 3. Fetch Real-Time Data from OpenWeather
# ─────────────────────────────────────────────────────────────
LAT = 30.1575
LON = 71.5249
URL = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"

print("Fetching Fresh Hourly Real-Time Data from OpenWeather...")
response = requests.get(URL)

if response.status_code != 200:
    raise Exception(f"API Failed with status code {response.status_code}: {response.text}")

data = response.json()

# ─────────────────────────────────────────────────────────────
# 4. Normalize and Clean the Feature Vector
# ─────────────────────────────────────────────────────────────
df = pd.json_normalize(data["list"])
df.columns = df.columns.str.replace("components.", "", regex=False)
df.columns = df.columns.str.replace("main.", "", regex=False)

df["city"]      = "Multan"
df["timestamp"] = pd.to_datetime(datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))

core_columns      = ["city", "timestamp", "aqi", "pm2_5", "pm10", "no2", "so2", "o3", "co"]
available_columns = [col for col in core_columns if col in df.columns]
df = df[available_columns]

# ─────────────────────────────────────────────────────────────
# 5. Replace OpenWeather 1–5 AQI with proper EPA AQI (0–500)
# ─────────────────────────────────────────────────────────────
print(f"OpenWeather raw AQI (1-5 scale): {df['aqi'].values[0]}")
print(f"Raw pollutants → PM2.5:{df['pm2_5'].values[0]:.1f} PM10:{df['pm10'].values[0]:.1f} "
      f"NO2:{df['no2'].values[0]:.1f} O3:{df['o3'].values[0]:.1f} "
      f"SO2:{df['so2'].values[0]:.1f} CO:{df['co'].values[0]:.1f} μg/m³")

df["aqi"] = df.apply(lambda row: calculate_epa_aqi(
    pm25=row.get("pm2_5"),
    pm10=row.get("pm10"),
    no2 =row.get("no2"),
    o3  =row.get("o3"),
    so2 =row.get("so2"),
    co  =row.get("co"),
), axis=1)

print(f"EPA AQI calculated from pollutants: {df['aqi'].values[0]}")
print("\nNew Feature Vector Extracted:")
print(df)

# ─────────────────────────────────────────────────────────────
# 6. Stream to BigQuery
# ─────────────────────────────────────────────────────────────
PROJECT_ID = "pearl-aqi-predictor"
REGION     = "us-central1"
DATASET_ID = "aqi_feature_store"
TABLE_ID   = "multan_historical"
table_path = f"{DATASET_ID}.{TABLE_ID}"

print("\nConnecting to Google Cloud Platform via Application Default Credentials...")
aiplatform.init(project=PROJECT_ID, location=REGION)
bq_client = bigquery.Client(project=PROJECT_ID)

print(f"Streaming to GCP table: {table_path}...")
pandas_gbq.to_gbq(
    df,
    destination_table=table_path,
    project_id=PROJECT_ID,
    if_exists="append"
)

print(f"Success! EPA AQI={df['aqi'].values[0]} appended to BigQuery.")
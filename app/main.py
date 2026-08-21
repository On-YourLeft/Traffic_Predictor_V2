from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
from typing import List, Optional
import joblib
import pandas as pd
import os
import math
from datetime import datetime, timedelta
import pytz
import requests
import time
import numpy as np

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# Initialize the FastAPI app
app = FastAPI(title="Smart Traffic Predictor")

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- DATABASE CONFIGURATION ---
DB_DIR = os.path.join(BASE_DIR, "../database")
os.makedirs(DB_DIR, exist_ok=True)
LOCAL_DB_PATH = os.path.abspath(os.path.join(DB_DIR, "local_telemetry.db"))

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{LOCAL_DB_PATH}")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class InferenceLog(Base):
    __tablename__ = "inference_logs"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    start_zone = Column(String)
    end_zone = Column(String)
    distance_km = Column(Float)
    ai_predicted_mins = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)

app.mount(
    "/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static"
)

# Load the saved Fused models
MODEL_PATH = os.path.join(BASE_DIR, "../models/vanet_rf_model.joblib")
COLUMNS_PATH = os.path.join(BASE_DIR, "../models/vanet_model_columns.joblib")

model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)


# Models for Request
class RouteSegment(BaseModel):
    model_config = ConfigDict(extra="allow")
    avg_speed_kmph: float
    distance_km: float
    # These might come from the frontend, but we don't strictly rely on them anymore
    density_veh_per_km: float = 0.0
    temp_c: float = 25.0
    rain_intensity_mmph: float = 0.0


class BatchRouteInput(BaseModel):
    segments: List[RouteSegment]
    departure_time: Optional[str] = (
        None  # <--- Fix is here # Replaces future_offset_mins
    )


class TelemetryInput(BaseModel):
    session_id: str
    start_zone: str
    end_zone: str
    distance_km: float
    ai_predicted_mins: float


def save_log_to_db(data: TelemetryInput):
    db = SessionLocal()
    try:
        db_log = InferenceLog(**data.model_dump())
        db.add(db_log)
        db.commit()
    except Exception as e:
        print(f"DB Error: {e}")
    finally:
        db.close()


@app.post("/log_telemetry")
def log_telemetry(data: TelemetryInput, background_tasks: BackgroundTasks):
    background_tasks.add_task(save_log_to_db, data)
    return {"status": "Telemetry logged"}


@app.get("/")
def read_root():
    return FileResponse(os.path.join(BASE_DIR, "static/index.html"))


import requests
from fastapi import HTTPException


import requests
from fastapi import HTTPException


@app.get("/api/pois")
def fetch_pois_proxy(amenity: str, bbox: str):
    # 1. Cap the search to 100 items so the Render IP never gets blacklisted for data dumping
    query = f"[out:json][timeout:60];nwr[amenity={amenity}]({bbox});out center 100;"

    # 2. Use the lz4 high-capacity load balancer built for heavy traffic
    url = "https://lz4.overpass-api.de/api/interpreter"

    # 3. CRITICAL: Forge a custom User-Agent so Overpass trusts the Render cloud IP
    headers = {
        "User-Agent": "DelhiTransitEngine/2.0 (traffic-engine-v2.onrender.com)",
        "Accept": "application/json",
    }

    try:
        # 4. Send as raw encoded bytes to bypass any 406 format rejections
        response = requests.post(
            url, data=query.encode("utf-8"), headers=headers, timeout=65
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ OVERPASS CRASH: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Backend proxy failed to communicate with Overpass API.",
        )


# --- LIVE ENVIRONMENTAL TELEMETRY ---
AQI_CACHE = {"penalty": 1.0, "last_fetched": 0, "severity": 3.0}


def get_live_delhi_environmental_penalty():
    """
    Fetches real-time weather and maps it to the new
    `environmental_friction_penalty` feature our model expects.
    """
    current_time = time.time()
    if current_time - AQI_CACHE["last_fetched"] < 3600:
        return AQI_CACHE["penalty"], AQI_CACHE["severity"]

    try:
        # We fetch current weather code (WMO) to detect Rain/Fog
        url = "https://api.open-meteo.com/v1/forecast?latitude=28.6139&longitude=77.2090&current_weather=true"
        response = requests.get(url, timeout=3)
        weather_code = response.json()["current_weather"]["weathercode"]

        # Mapping WMO codes to our friction penalty
        # 45, 48 are Fog/Smog. 50+ is Rain.
        if weather_code in [45, 48]:
            penalty = 1.45  # Severe Visibility
            severity = 5.0
        elif weather_code >= 50:
            penalty = 1.30  # Slippery Roads
            severity = 4.0
        else:
            penalty = 1.0  # Clear
            severity = 2.0

        AQI_CACHE["penalty"] = penalty
        AQI_CACHE["severity"] = severity
        AQI_CACHE["last_fetched"] = current_time
        return penalty, severity
    except Exception:
        return 1.0, 3.0


def build_live_feature_matrix(segment_distance, segment_speed, target_time):
    """
    Dynamically constructs the exact feature matrix.
    Now Calendar-Aware for precise date/time and festival surges!
    """
    feature_dict = {col: 0.0 for col in model_columns}

    if "distance_km" in feature_dict:
        feature_dict["distance_km"] = segment_distance
    if "average_speed_kmph" in feature_dict:
        feature_dict["average_speed_kmph"] = segment_speed

    # Extract exact temporal features from the Target Time
    hour = target_time.hour
    day = target_time.weekday()
    month = target_time.month
    day_of_month = target_time.day

    if "hour_sin" in feature_dict:
        feature_dict["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    if "hour_cos" in feature_dict:
        feature_dict["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    if "day_sin" in feature_dict:
        feature_dict["day_sin"] = np.sin(2 * np.pi * day / 7.0)
    if "day_cos" in feature_dict:
        feature_dict["day_cos"] = np.cos(2 * np.pi * day / 7.0)

    # --- 🎆 FESTIVAL & HOLIDAY CALIBRATOR ---
    is_festival = False
    # Example hardcoded dates (e.g., Aug 15th Independence Day, Nov Diwali season)
    if (month == 8 and day_of_month == 15) or (month == 11 and day_of_month <= 15):
        is_festival = True

    # Adjust AI multipliers based on calendar events
    surge = 1.45 if is_festival else 1.15
    if "historical_surge_multiplier" in feature_dict:
        feature_dict["historical_surge_multiplier"] = surge
    if "historical_wait_time" in feature_dict:
        feature_dict["historical_wait_time"] = 6.5 if is_festival else 4.2

    is_rush = 1 if (8 <= hour <= 11) or (17 <= hour <= 20) else 0
    if "vanet_avg_queue_length" in feature_dict:
        feature_dict["vanet_avg_queue_length"] = (
            15.5 if (is_rush or is_festival) else 6.2
        )
    if "vanet_comm_delay_ms" in feature_dict:
        feature_dict["vanet_comm_delay_ms"] = 65.0 if is_rush else 35.0

    if "route_total_stops" in feature_dict:
        feature_dict["route_total_stops"] = 12.0

    env_penalty, _ = get_live_delhi_environmental_penalty()
    if "environmental_friction_penalty" in feature_dict:
        feature_dict["environmental_friction_penalty"] = env_penalty

    return pd.DataFrame([feature_dict])[model_columns]


# --- CORE PREDICTION ENDPOINT ---
@app.post("/predict_route_segments")
# --- CORE PREDICTION ENDPOINT ---
@app.post("/predict_route_segments")
def predict_route_segments(data: BatchRouteInput):
    ist = pytz.timezone("Asia/Kolkata")

    # Check if a custom date/time was sent, otherwise default to NOW
    if data.departure_time:
        # Parse ISO string from JS and localize to IST
        target_time = datetime.fromisoformat(data.departure_time.replace("Z", "+00:00"))
        if target_time.tzinfo is None:
            target_time = ist.localize(target_time)
        else:
            target_time = target_time.astimezone(ist)
    else:
        target_time = datetime.now(ist)

    try:
        _, live_aqi_severity = get_live_delhi_environmental_penalty()

        # Calculate totals for the whole trip
        total_distance = sum(seg.distance_km for seg in data.segments)
        avg_speed = (
            sum(seg.avg_speed_kmph * seg.distance_km for seg in data.segments)
            / total_distance
            if total_distance > 0
            else 30.0
        )

        # 1. Build the advanced feature matrix using the exact TARGET TIME
        macro_features = build_live_feature_matrix(
            segment_distance=total_distance,
            segment_speed=avg_speed,
            target_time=target_time,
        )

        # 2. Predict
        total_predicted_mins = float(model.predict(macro_features)[0])
        safe_total_mins = max(1.0, total_predicted_mins)

        # 3. Micro-Segment Distribution
        segment_base_times = [
            (seg.distance_km / max(seg.avg_speed_kmph, 1.0)) for seg in data.segments
        ]
        sum_base_times = sum(segment_base_times)

        predictions = [
            (safe_total_mins * (base_time / max(sum_base_times, 0.0001)))
            for base_time in segment_base_times
        ]

        return {
            "segment_predictions": predictions,
            "system_time": {
                "hour": target_time.hour,
                "is_weekend": 1 if target_time.weekday() >= 5 else 0,
                "is_festival": (
                    1
                    if (target_time.month == 8 and target_time.day == 15)
                    or (target_time.month == 11 and target_time.day <= 15)
                    else 0
                ),
            },
            "environmental_telemetry": {"live_aqi_severity": live_aqi_severity},
        }

    except Exception as e:
        print(f"🔥 [CRITICAL API ERROR CAUGHT]: {e}")
        # Failsafe logic remains the same...

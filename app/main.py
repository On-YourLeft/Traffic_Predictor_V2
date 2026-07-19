from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
from typing import List
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

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
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

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Load the saved Fused models
MODEL_PATH = os.path.join(BASE_DIR, "../models/vanet_rf_model.joblib")
COLUMNS_PATH = os.path.join(BASE_DIR, "../models/vanet_model_columns.joblib")

model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)

# Models for Request
class RouteSegment(BaseModel):
    model_config = ConfigDict(extra='allow')
    avg_speed_kmph: float
    distance_km: float
    # These might come from the frontend, but we don't strictly rely on them anymore
    density_veh_per_km: float = 0.0
    temp_c: float = 25.0
    rain_intensity_mmph: float = 0.0

class BatchRouteInput(BaseModel):
    segments: List[RouteSegment]
    future_offset_mins: int = 0

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
            penalty = 1.0   # Clear
            severity = 2.0
            
        AQI_CACHE["penalty"] = penalty
        AQI_CACHE["severity"] = severity
        AQI_CACHE["last_fetched"] = current_time
        return penalty, severity
    except Exception:
        return 1.0, 3.0

def build_live_feature_matrix(segment_distance, segment_speed, future_offset_mins):
    """
    Dynamically constructs the exact feature matrix the Utter Perfection model expects.
    """
    # 1. Create a blank dictionary of all zeros based on the exact columns the model was trained on
    feature_dict = {col: 0.0 for col in model_columns}
    
    # 2. Inject core routing telemetry
    if 'distance_km' in feature_dict:
        feature_dict['distance_km'] = segment_distance
    if 'average_speed_kmph' in feature_dict:
        feature_dict['average_speed_kmph'] = segment_speed
        
    # 3. Inject Real-Time Cyclical Math (With future slider support)
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist) + timedelta(minutes=future_offset_mins)
    hour = now.hour
    day = now.weekday()
    
    if 'hour_sin' in feature_dict: feature_dict['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    if 'hour_cos' in feature_dict: feature_dict['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    if 'day_sin' in feature_dict: feature_dict['day_sin'] = np.sin(2 * np.pi * day / 7.0)
    if 'day_cos' in feature_dict: feature_dict['day_cos'] = np.cos(2 * np.pi * day / 7.0)
        
    # 4. Inject Domain Fusion Defaults (City-wide medians)
    if 'historical_surge_multiplier' in feature_dict: feature_dict['historical_surge_multiplier'] = 1.15 
    if 'historical_wait_time' in feature_dict: feature_dict['historical_wait_time'] = 4.2 
    
    # Base queue lengths slightly higher if it's rush hour
    is_rush = 1 if (8 <= hour <= 11) or (17 <= hour <= 20) else 0
    if 'vanet_avg_queue_length' in feature_dict: feature_dict['vanet_avg_queue_length'] = 15.5 if is_rush else 6.2
    if 'vanet_comm_delay_ms' in feature_dict: feature_dict['vanet_comm_delay_ms'] = 65.0 if is_rush else 35.0
    
    # Default route stops to induce baseline urban friction
    if 'route_total_stops' in feature_dict: feature_dict['route_total_stops'] = 12.0 
    
    # 5. Inject Live Environmental Constraints
    env_penalty, _ = get_live_delhi_environmental_penalty()
    if 'environmental_friction_penalty' in feature_dict: 
        feature_dict['environmental_friction_penalty'] = env_penalty
        
    # 6. Convert to a DataFrame perfectly ordered for the Random Forest
    return pd.DataFrame([feature_dict])[model_columns]

# --- CORE PREDICTION ENDPOINT ---
@app.post("/predict_route_segments")
def predict_route_segments(data: BatchRouteInput):
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist) + timedelta(minutes=data.future_offset_mins)
    
    try:
        _, live_aqi_severity = get_live_delhi_environmental_penalty()
        predictions = []
        
        for seg in data.segments:
            # 1. Build the advanced feature matrix for this specific segment
            advanced_features = build_live_feature_matrix(
                segment_distance=seg.distance_km, 
                segment_speed=seg.avg_speed_kmph,
                future_offset_mins=data.future_offset_mins
            )
            
            # 2. Let the Utter Perfection model predict the time directly
            predicted_mins = model.predict(advanced_features)[0]
            
            # Prevent the AI from predicting negative time
            safe_mins = max(0.1, float(predicted_mins))
            predictions.append(safe_mins)

        return {
            "segment_predictions": list(predictions),
            "system_time": {"hour": now.hour, "is_weekend": 1 if now.weekday() >= 5 else 0},
            "environmental_telemetry": {"live_aqi_severity": live_aqi_severity}
        }

    except Exception as e:
        print(f"🔥 [CRITICAL API ERROR CAUGHT]: {e}")
        # Mathematical Fallback
        fallback_predictions = []
        for seg in data.segments:
            raw_time = (seg.distance_km / max(seg.avg_speed_kmph, 1.0)) * 60.0
            fallback_predictions.append(float(raw_time * 1.1))
            
        return {
            "segment_predictions": fallback_predictions,
            "system_time": {"hour": now.hour, "is_weekend": 0},
            "environmental_telemetry": {"live_aqi_severity": 3.0},
            "status": "failsafe_engaged"
        }
# 🚦 Delhi Transit Intelligence Engine

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg?logo=docker)](https://www.docker.com/)
[![Render](https://img.shields.io/badge/Render-Deployed-000000.svg?logo=render)](https://render.com/)
[![MLOps](https://img.shields.io/badge/MLOps-CI%2FCD-4CAF50.svg)]()

An autonomous, domain-adapted Machine Learning pipeline that predicts highly accurate ETAs, visualizes traffic telemetry, and provides intelligent routing for the Delhi NCR region.

Unlike standard routing apps that rely purely on static speed limits, this engine fuses **OSRM spatial geometries** with a **Random Forest ML model** trained on historical domain data (Uber historical velocities, VANET queue lengths, Metro hub proximities, and cyclical time metrics) to generate highly realistic urban friction models.

## ✨ Core Architecture & Features

* 🧠 **Macro-Prediction AI Architecture:** Bypasses the Random Forest "Leaf Node Extrapolation Trap" by predicting the ETA of the *entire* route macroscopically, then proportionally slicing the ETA down into micro-segments based on localized spatial variance and bottlenecks.
* ☁️ **Live Environmental & AQI Telemetry:** Fetches real-time weather and visibility data via Open-Meteo API, dynamically injecting an `environmental_friction_penalty` into the model to account for rain, smog, and fog delays.
* 📊 **MLflow Model Tracking:** Complete ML lifecycle management using MLflow to track hyperparameters, log model metrics, and monitor Data Drift across training iterations.
* 🗺️ **Dynamic Waze-Style Visualizer:** A deterministic UI engine maps synthetic telemetry into a segmented, multi-colored Leaflet map (Green/Orange/Red) based on the AI's predicted speed for that specific 300m stretch.
* ⏳ **Multi-Horizon Forecasting:** An interactive, custom-styled time-series slider that recalculates cyclical temporal sine/cosine features, allowing users to "time travel" and predict route congestion up to 120 minutes into the future.
* 📍 **Live GPS & Multi-Stop Pitstops:** HTML5 Geolocation API integration combined with dynamic OSRM waypoint sequencing allows users to route from their live location and seamlessly inject/drop multiple amenity stops into their path.
* ⛽ **Geospatial POI Radar (Turf.js):** Real-time spatial scanning using the Overpass API to detect optimal Fuel and Cafe stops strictly within a 2.5km deviation of the active route polygon.
* 🗄️ **Asynchronous Telemetry Database:** A background SQLAlchemy pipeline that silently logs anonymous session data, queried routes, and AI predictions to a PostgreSQL database for historical analysis.
* ⚙️ **Fully Automated CI/CD:** A complete MLOps pipeline using GitHub Actions and Docker Buildx (with layer caching) to automatically build, containerize, and deploy the Python/FastAPI backend to Render Cloud upon every push to the `main` branch.

## 🛠️ Tech Stack

* **Backend:** Python, FastAPI, Uvicorn, SQLAlchemy (PostgreSQL / SQLite)
* **Machine Learning:** Scikit-Learn (Random Forest max_depth=18), MLflow, Pandas, NumPy, Joblib
* **Frontend:** HTML5, CSS3, Vanilla JavaScript, Leaflet.js, Turf.js
* **DevOps/Infrastructure:** Docker, Docker Hub, GitHub Actions, Render Cloud
* **External APIs:** OSRM (Routing), Open-Meteo (Weather), Overpass API (OSM POIs), Nominatim (Geocoding)

## 🚀 Local Development Setup

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/traffic_predictor.git
cd traffic_predictor

```

**2. Create a Virtual Environment & Install Dependencies**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

```

**3. Run the FastAPI Server**

```bash
uvicorn app.main:app --reload

```

*(Note: Running locally will automatically generate an isolated `database/local_telemetry.db` SQLite file for telemetry tracking).*

Navigate to `http://localhost:8000` in your browser to view the application.

## 🐳 Docker & CI/CD Deployment

This application is fully containerized and features a zero-touch CI/CD deployment pipeline.

### Environment Variables & Secrets

To utilize the GitHub Actions pipeline, configure the following secrets in your GitHub repository (`Settings > Secrets and variables > Actions`):

* `DOCKER_USERNAME`: Your Docker Hub username.
* `DOCKER_PASSWORD`: Your Docker Hub Access Token.
* `RENDER_DEPLOY_HOOK_URL`: The unique Webhook URL from your Render Web Service.

**Render Cloud Environment Variables:**

* `DATABASE_URL`: The Internal PostgreSQL connection string (e.g., `postgresql://user:pass@host/dbname`) to persist telemetry logs across container restarts.

### Deployment Flow

1. Push code to the `main` branch.
2. GitHub Actions initializes **Docker Buildx**.
3. The pipeline fetches cached layers, installs Python dependencies, and forces the inclusion of the `.joblib` ML artifacts.
4. The new image is pushed to Docker Hub (`YOUR_USERNAME/traffic-predictor:latest`).
5. GitHub fires a POST request to Render, seamlessly rolling over the live server to the new container.

## 📈 Future Roadmap (Phase 4)

* **Multi-City Scaling**: Expand the geospatial boundary boxes and training data to include Mumbai, Bengaluru, and Pune.
* **Live CCTV Computer Vision**: Integrate edge-deployed YOLOv8 models to count vehicles from public traffic cameras and inject live density metrics directly into the Random Forest.
* **Kubernetes Orchestration**: Transition from a single container deployment to a Kubernetes cluster for horizontal auto-scaling during high-traffic API requests.

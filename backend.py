from flask import Flask, request, jsonify
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, Column, Float, Integer, String, Date
from sqlalchemy.orm import declarative_base, sessionmaker
import joblib
from flask_cors import CORS
import requests
import logging
import sys
import numpy as np

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ---------------- Flask App ----------------
app = Flask(__name__)
CORS(app)

# ---------------- Load Model ----------------
try:
    model = joblib.load("model.pkl")
    if not hasattr(model, "feature_names_in_"):
        raise AttributeError("Model missing attribute 'feature_names_in_'")
    model.feature_names_in_ = [str(col) for col in model.feature_names_in_]
    logging.info("✅ Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise RuntimeError(f"Failed to load model: {e}")

# ---------------- Database ----------------
Base = declarative_base()

class EnergyPrediction(Base):
    __tablename__ = "energy_prediction"
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date)
    time = Column(String(8))  # HH:MM:SS
    city = Column(String(50))
    temperature = Column(Float)
    month = Column(Integer)
    day = Column(Integer)
    dayofweek = Column(Integer)
    dayofyear = Column(Integer)
    hour = Column(Integer)  # selected hour
    predicted_consumption = Column(Float)
    solar_contribution = Column(Float)
    net_demand = Column(Float)

engine = create_engine("sqlite:///energy_prediction.db")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# ---------------- Weather API ----------------
API_KEY = "331d6fab83c069eeb8b418f77f90f5fd"
DEFAULT_TEMP = 30

def get_temperature(city: str) -> float:
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        if resp.status_code == 200:
            return float(data["main"]["temp"])
        logging.warning(f"Weather API returned error for {city}: {data}")
    except Exception as e:
        logging.warning(f"Weather fetch failed for {city}: {e}")
    return DEFAULT_TEMP

# ---------------- Trained Cities ----------------
TRAINED_CITIES = [col.split("_")[1] for col in model.feature_names_in_ if col.startswith("City_")]

# ---------------- Routes ----------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    logging.info(f"📩 Received request: {data}")

    if not data:
        return jsonify({"error": "No data provided"}), 400

    city = str(data.get("city", "")).strip().title()
    date_str = data.get("date")
    time_str = data.get("time")  # Expect format HH:MM

    if not city or not date_str or not time_str:
        return jsonify({"error": "Missing 'city', 'date', or 'time'"}), 400

    if city not in TRAINED_CITIES:
        return jsonify({"error": f"City '{city}' not in trained list", "valid_cities": TRAINED_CITIES}), 400

    try:
        date_obj = pd.to_datetime(date_str)
        hour = int(time_str.split(":")[0])
    except Exception:
        return jsonify({"error": "Invalid date or time format"}), 400

    # Fetch temperature
    temperature = get_temperature(city)

    # Prepare features
    city_features = {f"City_{c}": 0 for c in TRAINED_CITIES}
    city_features[f"City_{city}"] = 1
    features = {
        "month": date_obj.month,
        "day": date_obj.day,
        "dayofweek": date_obj.dayofweek,
        "dayofyear": date_obj.dayofyear,
        "hour": hour,
        "Temperature (°C)": temperature,
        **city_features,
    }

    features_df = pd.DataFrame([features])
    for col in model.feature_names_in_:
        if col not in features_df.columns:
            features_df[col] = 0
    features_df = features_df[model.feature_names_in_]

    # Predict
    try:
        predicted_consumption = max(0, float(model.predict(features_df)[0]))
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    # Simulate solar contribution
    solar_contribution = 0
    if 6 <= hour <= 18:
        solar_contribution = round(np.random.uniform(0.1, 0.7) * predicted_consumption, 2)
    net_demand = max(0, predicted_consumption - solar_contribution)

    # Save to DB
    db = Session()
    try:
        new_prediction = EnergyPrediction(
            date=date_obj.date(),
            time=f"{hour}:00:00",
            city=city,
            temperature=temperature,
            month=date_obj.month,
            day=date_obj.day,
            dayofweek=date_obj.dayofweek,
            dayofyear=date_obj.dayofyear,
            hour=hour,
            predicted_consumption=predicted_consumption,
            solar_contribution=solar_contribution,
            net_demand=net_demand
        )
        db.add(new_prediction)
        db.commit()
    except Exception as e:
        db.rollback()
        logging.error(f"Database error: {e}")
        return jsonify({"error": f"Database error: {e}"}), 500
    finally:
        db.close()

    return jsonify({
        "date": date_obj.strftime("%Y-%m-%d"),
        "time": f"{hour}:00",
        "city": city,
        "temperature": temperature,
        "Total_Consumption_kWh": round(predicted_consumption, 2),
        "Solar_Contribution_kWh": round(solar_contribution, 2),
        "Net_Demand_kWh": round(net_demand, 2),
        "status": "Prediction successful and saved to database"
    })

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "database_connected": engine is not None,
        "trained_cities": TRAINED_CITIES,
        "server_time": datetime.now().isoformat(),
    })

# ---------------- Main ----------------
if __name__ == "__main__":
    logging.info("Server running at http://127.0.0.1:5001")
    app.run(host="0.0.0.0", port=5001, debug=True)
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="PM2.5 Prediction API")

MODEL_FILE = "ml/pm25_model_v2.pkl"

model = joblib.load(MODEL_FILE)


class PredictionInput(BaseModel):
    temperature_2m: float
    relative_humidity_2m: float
    pressure_msl: float
    precipitation: float
    wind_speed_10m: float
    pm25_lag_1: float
    pm25_lag_3: float
    pm25_rolling_mean_3: float
    pm25_rolling_mean_6: float


@app.get("/")
def home():
    return {
        "message": "PM2.5 Prediction API is running"
    }


@app.post("/predict")
def predict(data: PredictionInput):

    input_data = pd.DataFrame([{
        "temperature_2m": data.temperature_2m,
        "relative_humidity_2m": data.relative_humidity_2m,
        "pressure_msl": data.pressure_msl,
        "precipitation": data.precipitation,
        "wind_speed_10m": data.wind_speed_10m,
        "pm25_lag_1": data.pm25_lag_1,
        "pm25_lag_3": data.pm25_lag_3,
        "pm25_rolling_mean_3": data.pm25_rolling_mean_3,
        "pm25_rolling_mean_6": data.pm25_rolling_mean_6
    }])

    prediction = model.predict(input_data)[0]

    return {
        "predicted_pm25": round(float(prediction), 2)
    }
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("ml/pm25_model.pkl")

# Example weather conditions
weather_data = pd.DataFrame({
    "temperature_2m": [20],
    "relative_humidity_2m": [60],
    "pressure_msl": [1015],
    "precipitation": [0],
    "wind_speed_10m": [2]
})

# Make prediction
prediction = model.predict(weather_data)

print("Predicted PM2.5:", round(prediction[0], 2))

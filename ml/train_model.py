import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# -----------------------------
# 1. Load the dataset
# -----------------------------
file_path = "data/cleaned/air_quality_weather_joined.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Number of rows:", len(df))

# -----------------------------
# 2. Select features
# -----------------------------
features = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "precipitation",
    "wind_speed_10m"
]

target = "value"

X = df[features]
y = df[target]

# Remove rows with missing values
data = pd.concat([X, y], axis=1).dropna()

X = data[features]
y = data[target]

print("Rows after cleaning:", len(data))

# -----------------------------
# 3. Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# -----------------------------
# 4. Create and train model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model trained successfully!")

# -----------------------------
# 5. Make predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 6. Evaluate model
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("-------------------------")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 2))

# -----------------------------
# 7. Save the trained model
# -----------------------------
model_path = "ml/pm25_model.pkl"

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("File:", os.path.abspath(model_path))
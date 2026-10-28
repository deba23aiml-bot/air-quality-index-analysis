import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


INPUT_FILE = "data/cleaned/air_quality_features.csv"


# Load feature dataset
df = pd.read_csv(INPUT_FILE)

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort chronologically
df = df.sort_values("timestamp").reset_index(drop=True)

print("Feature dataset loaded successfully!")
print(f"Total rows: {len(df)}")


# Features
features = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "precipitation",
    "wind_speed_10m",
    "pm25_lag_1",
    "pm25_lag_3",
    "pm25_rolling_mean_3",
    "pm25_rolling_mean_6"
]

target = "value"

X = df[features]
y = df[target]


# Chronological split
# 60% Training
# 20% Validation
# 20% Testing

train_end = int(len(df) * 0.60)
validation_end = int(len(df) * 0.80)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_validation = X.iloc[train_end:validation_end]
y_validation = y.iloc[train_end:validation_end]

X_test = X.iloc[validation_end:]
y_test = y.iloc[validation_end:]


print("\nChronological Data Split")
print("-------------------------")
print(f"Training rows   : {len(X_train)}")
print(f"Validation rows : {len(X_validation)}")
print(f"Testing rows    : {len(X_test)}")


# Train Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

import joblib

joblib.dump(model, "ml/pm25_model_v2.pkl")
print("Model saved as ml/pm25_model_v2.pkl")

print("\nModel trained successfully!")


# Validation prediction
validation_predictions = model.predict(X_validation)

validation_mae = mean_absolute_error(
    y_validation,
    validation_predictions
)

validation_rmse = mean_squared_error(
    y_validation,
    validation_predictions
) ** 0.5


print("\nValidation Performance")
print("-------------------------")
print(f"MAE  : {validation_mae:.2f}")
print(f"RMSE : {validation_rmse:.2f}")


# Final test prediction
test_predictions = model.predict(X_test)

test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_rmse = mean_squared_error(
    y_test,
    test_predictions
) ** 0.5

test_r2 = r2_score(
    y_test,
    test_predictions
)


print("\nTest Performance")
print("-------------------------")
print(f"MAE  : {test_mae:.2f}")
print(f"RMSE : {test_rmse:.2f}")
print(f"R²   : {test_r2:.2f}")
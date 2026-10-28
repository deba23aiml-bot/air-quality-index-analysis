import mlflow
import mlflow.sklearn

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


INPUT_FILE = "data/cleaned/air_quality_features.csv"


# Load data
df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)


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


# Chronological 60/20/20 split
train_end = int(len(df) * 0.60)
validation_end = int(len(df) * 0.80)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_test = X.iloc[validation_end:]
y_test = y.iloc[validation_end:]


# MLflow experiment
mlflow.set_experiment("PM25_Prediction")


with mlflow.start_run(run_name="RandomForest_Regression"):

    # Model parameters
    n_estimators = 100
    random_state = 42

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, predictions)

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(y_test, predictions)


    # Log parameters
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("train_rows", len(X_train))
    mlflow.log_param("test_rows", len(X_test))


    # Log metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)


    # Log model artifact
    mlflow.sklearn.log_model(
        model,
        name="pm25_random_forest_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )


    print("MLflow experiment completed successfully!")

    print("\nLogged Parameters")
    print("-------------------------")
    print(f"Model       : RandomForestRegressor")
    print(f"Estimators  : {n_estimators}")
    print(f"Train rows  : {len(X_train)}")
    print(f"Test rows   : {len(X_test)}")

    print("\nLogged Metrics")
    print("-------------------------")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.2f}")
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


INPUT_FILE = "data/cleaned/air_quality_features.csv"


# Load dataset
df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort chronologically
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


# Create categories
def classify_pm25(value):
    if value < 60:
        return "Good"
    elif value < 120:
        return "Moderate"
    elif value < 250:
        return "Poor"
    else:
        return "Very Poor"


df["category"] = df["value"].apply(classify_pm25)


X = df[features]


# Chronological split
train_end = int(len(df) * 0.60)
validation_end = int(len(df) * 0.80)


X_train = X.iloc[:train_end]
X_test = X.iloc[validation_end:]


# =====================================================
# REGRESSION MODEL
# =====================================================

y_reg = df["value"]

y_reg_train = y_reg.iloc[:train_end]
y_reg_test = y_reg.iloc[validation_end:]


regression_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

regression_model.fit(X_train, y_reg_train)

regression_predictions = regression_model.predict(X_test)


reg_mae = mean_absolute_error(
    y_reg_test,
    regression_predictions
)

reg_rmse = mean_squared_error(
    y_reg_test,
    regression_predictions
) ** 0.5

reg_r2 = r2_score(
    y_reg_test,
    regression_predictions
)


# =====================================================
# CLASSIFICATION MODEL
# =====================================================

y_class = df["category"]

y_class_train = y_class.iloc[:train_end]
y_class_test = y_class.iloc[validation_end:]


classification_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

classification_model.fit(X_train, y_class_train)

classification_predictions = classification_model.predict(X_test)


class_accuracy = accuracy_score(
    y_class_test,
    classification_predictions
)

class_precision = precision_score(
    y_class_test,
    classification_predictions,
    average="weighted",
    zero_division=0
)

class_recall = recall_score(
    y_class_test,
    classification_predictions,
    average="weighted",
    zero_division=0
)

class_f1 = f1_score(
    y_class_test,
    classification_predictions,
    average="weighted",
    zero_division=0
)


# =====================================================
# DISPLAY COMPARISON
# =====================================================

print("\n==========================================")
print("       MODEL COMPARISON")
print("==========================================")

print("\nRegression Model")
print("------------------------------------------")
print(f"MAE  : {reg_mae:.2f}")
print(f"RMSE : {reg_rmse:.2f}")
print(f"R²   : {reg_r2:.2f}")

print("\nClassification Model")
print("------------------------------------------")
print(f"Accuracy  : {class_accuracy:.2f}")
print(f"Precision : {class_precision:.2f}")
print(f"Recall    : {class_recall:.2f}")
print(f"F1 Score  : {class_f1:.2f}")

print("\n==========================================")
print("Comparison completed successfully!")
print("==========================================")
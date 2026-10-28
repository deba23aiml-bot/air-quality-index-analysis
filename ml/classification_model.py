import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


INPUT_FILE = "data/cleaned/air_quality_features.csv"


# Load dataset
df = pd.read_csv(INPUT_FILE)

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort chronologically
df = df.sort_values("timestamp").reset_index(drop=True)


# Create PM2.5 categories
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

X = df[features]
y = df["category"]


# Chronological 60/20/20 split
train_end = int(len(df) * 0.60)
validation_end = int(len(df) * 0.80)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_validation = X.iloc[train_end:validation_end]
y_validation = y.iloc[train_end:validation_end]

X_test = X.iloc[validation_end:]
y_test = y.iloc[validation_end:]


print("Classification Dataset")
print("-------------------------")
print(f"Training rows   : {len(X_train)}")
print(f"Validation rows : {len(X_validation)}")
print(f"Testing rows    : {len(X_test)}")


# Train classification model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nClassification model trained successfully!")


# Validation prediction
validation_predictions = model.predict(X_validation)

validation_accuracy = accuracy_score(
    y_validation,
    validation_predictions
)

validation_precision = precision_score(
    y_validation,
    validation_predictions,
    average="weighted",
    zero_division=0
)

validation_recall = recall_score(
    y_validation,
    validation_predictions,
    average="weighted",
    zero_division=0
)

validation_f1 = f1_score(
    y_validation,
    validation_predictions,
    average="weighted",
    zero_division=0
)


print("\nValidation Performance")
print("-------------------------")
print(f"Accuracy  : {validation_accuracy:.2f}")
print(f"Precision : {validation_precision:.2f}")
print(f"Recall    : {validation_recall:.2f}")
print(f"F1 Score  : {validation_f1:.2f}")


# Test prediction
test_predictions = model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)


print("\nTest Performance")
print("-------------------------")
print(f"Accuracy  : {test_accuracy:.2f}")
print(f"Precision : {test_precision:.2f}")
print(f"Recall    : {test_recall:.2f}")
print(f"F1 Score  : {test_f1:.2f}")
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import time
import os
import json
from datetime import datetime

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Air Quality Index Analysis",
    page_icon="🌍",
    layout="wide"
)

DATA_FILE = "data/cleaned/air_quality_weather_joined.csv"
FEATURE_FILE = "data/cleaned/air_quality_features.csv"
PREDICTION_LOG = "ml/prediction_log.csv"
REGISTRY_FILE = "ml/model_registry.json"

API_URL = "http://127.0.0.1:8000/predict"
API_DOCS_URL = "http://127.0.0.1:8000/docs"

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🌍 Air Quality Index Analysis and Prediction")
st.write("Data Engineering and MLOps Project")
st.divider()

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    st.error(f"Dataset not found: {DATA_FILE}")
    st.info("Run the app from your project root folder.")
    st.stop()
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()

required_columns = ["timestamp", "pollutant", "value"]
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(f"Required columns missing: {missing_columns}")
    st.stop()

df["timestamp"] = pd.to_datetime(
    df["timestamp"], errors="coerce"
)
df["value"] = pd.to_numeric(
    df["value"], errors="coerce"
)

df = df.dropna(
    subset=["timestamp", "pollutant", "value"]
)

df = df.sort_values("timestamp").reset_index(drop=True)

if df.empty:
    st.error("No valid records are available.")
    st.stop()

# ---------------------------------------------------------
# SIDEBAR FILTER
# ---------------------------------------------------------

st.sidebar.header("Filters")

pollutants = sorted(
    df["pollutant"].dropna().unique().tolist()
)

selected_pollutant = st.sidebar.selectbox(
    "Select Pollutant", pollutants
)

filtered_df = df[
    df["pollutant"] == selected_pollutant
].copy()

if filtered_df.empty:
    st.warning("No measurements are available.")
    st.stop()

# ---------------------------------------------------------
# PM2.5 CATEGORY
# ---------------------------------------------------------

def categorize_pm25(value):
    if value < 60:
        return "Good"
    elif value < 120:
        return "Moderate"
    elif value < 250:
        return "Poor"
    else:
        return "Very Poor"


filtered_df["category"] = filtered_df["value"].apply(
    categorize_pm25
)

# ---------------------------------------------------------
# AIR QUALITY OVERVIEW
# ---------------------------------------------------------

st.header("Air Quality Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Measurements", f"{len(filtered_df):,}")

with col2:
    st.metric(
        "Average PM2.5",
        f"{filtered_df['value'].mean():.2f} µg/m³"
    )

with col3:
    st.metric(
        "Minimum PM2.5",
        f"{filtered_df['value'].min():.2f} µg/m³"
    )

with col4:
    st.metric(
        "Maximum PM2.5",
        f"{filtered_df['value'].max():.2f} µg/m³"
    )

st.caption(
    "PM2.5 category thresholds used in this project are "
    "Good (<60), Moderate (60–119.99), Poor (120–249.99), "
    "and Very Poor (250 and above)."
)

# ---------------------------------------------------------
# VISUALIZATION 1: PM2.5 OVER TIME
# ---------------------------------------------------------

st.header("Data Visualizations")

st.subheader("1. PM2.5 Concentration Over Time")

fig1 = px.line(
    filtered_df.sort_values("timestamp"),
    x="timestamp",
    y="value",
    title="PM2.5 Concentration Over Time",
    labels={
        "timestamp": "Time",
        "value": "PM2.5 (µg/m³)"
    }
)

st.plotly_chart(fig1, use_container_width=True)

# ---------------------------------------------------------
# VISUALIZATION 2: PM2.5 VS TEMPERATURE
# ---------------------------------------------------------

st.subheader("2. PM2.5 vs Temperature")

if "temperature_2m" in filtered_df.columns:
    temperature_df = filtered_df.dropna(
        subset=["temperature_2m", "value"]
    )

    fig2 = px.scatter(
        temperature_df,
        x="temperature_2m",
        y="value",
        title="PM2.5 vs Temperature",
        labels={
            "temperature_2m": "Temperature (°C)",
            "value": "PM2.5 (µg/m³)"
        }
    )

    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Temperature data is not available.")

# ---------------------------------------------------------
# VISUALIZATION 3: CATEGORY DISTRIBUTION
# ---------------------------------------------------------

st.subheader("3. PM2.5 Category Distribution")

category_counts = (
    filtered_df["category"]
    .value_counts()
    .reindex(
        ["Good", "Moderate", "Poor", "Very Poor"],
        fill_value=0
    )
    .rename_axis("Category")
    .reset_index(name="Count")
)

fig3 = px.bar(
    category_counts,
    x="Category",
    y="Count",
    title="PM2.5 Category Distribution",
    labels={
        "Category": "PM2.5 Category",
        "Count": "Number of Measurements"
    }
)

st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------
# VISUALIZATION 4: PM2.5 VALUE DISTRIBUTION
# ---------------------------------------------------------

st.subheader("4. PM2.5 Value Distribution")

fig4 = px.histogram(
    filtered_df,
    x="value",
    nbins=30,
    title="Distribution of PM2.5 Values",
    labels={
        "value": "PM2.5 (µg/m³)",
        "count": "Number of Measurements"
    }
)

st.plotly_chart(fig4, use_container_width=True)

# ---------------------------------------------------------
# VISUALIZATION 5: PM2.5 VS HUMIDITY
# ---------------------------------------------------------

st.subheader("5. PM2.5 vs Relative Humidity")

if "relative_humidity_2m" in filtered_df.columns:
    humidity_df = filtered_df.dropna(
        subset=["relative_humidity_2m", "value"]
    )

    fig5 = px.scatter(
        humidity_df,
        x="relative_humidity_2m",
        y="value",
        title="PM2.5 vs Relative Humidity",
        labels={
            "relative_humidity_2m": "Relative Humidity (%)",
            "value": "PM2.5 (µg/m³)"
        }
    )

    st.plotly_chart(fig5, use_container_width=True)
else:
    st.info("Humidity data is not available.")

# ---------------------------------------------------------
# PART 2: PM2.5 PREDICTION
# ---------------------------------------------------------

st.header("PM2.5 Prediction")

st.write(
    "Enter weather and historical PM2.5 values to predict "
    "the PM2.5 concentration."
)

# ---------------------------------------------------------
# LOAD FEATURE DATA
# ---------------------------------------------------------

try:
    feature_df = pd.read_csv(FEATURE_FILE)

    if "timestamp" in feature_df.columns:
        feature_df["timestamp"] = pd.to_datetime(
            feature_df["timestamp"],
            errors="coerce"
        )
        feature_df = feature_df.dropna(
            subset=["timestamp"]
        )

except FileNotFoundError:
    feature_df = None
    st.warning(
        "Feature dataset was not found. Prediction requires "
        "a running API and a compatible trained model."
    )
except Exception as e:
    feature_df = None
    st.warning(f"Could not load feature data: {e}")

# ---------------------------------------------------------
# PREDICTION INPUTS
# ---------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    temperature = st.number_input(
        "Temperature (°C)", value=25.0
    )
    humidity = st.number_input(
        "Relative Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=60.0
    )
    pressure = st.number_input(
        "Pressure (hPa)",
        min_value=0.0,
        value=1010.0
    )

with col2:
    precipitation = st.number_input(
        "Precipitation (mm)",
        min_value=0.0,
        value=0.0
    )
    wind_speed = st.number_input(
        "Wind Speed (km/h)",
        min_value=0.0,
        value=5.0
    )
    lag_1 = st.number_input(
        "PM2.5 Lag-1",
        min_value=0.0,
        value=150.0
    )

with col3:
    lag_3 = st.number_input(
        "PM2.5 Lag-3",
        min_value=0.0,
        value=150.0
    )
    rolling_3 = st.number_input(
        "PM2.5 Rolling Mean-3",
        min_value=0.0,
        value=150.0
    )
    rolling_6 = st.number_input(
        "PM2.5 Rolling Mean-6",
        min_value=0.0,
        value=150.0
    )

# ---------------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------------

if st.button("Predict PM2.5", type="primary"):

    input_data = {
        "temperature_2m": temperature,
        "relative_humidity_2m": humidity,
        "pressure_msl": pressure,
        "precipitation": precipitation,
        "wind_speed_10m": wind_speed,
        "pm25_lag_1": lag_1,
        "pm25_lag_3": lag_3,
        "pm25_rolling_mean_3": rolling_3,
        "pm25_rolling_mean_6": rolling_6
    }

    start_time = time.time()

    try:
        response = requests.post(
            API_URL,
            json=input_data,
            timeout=10
        )

        latency = (time.time() - start_time) * 1000

        if response.status_code == 200:
            result = response.json()

            if "predicted_pm25" not in result:
                st.error(
                    "API response is missing predicted_pm25. "
                    "Check ml/api.py."
                )
            else:
                prediction = float(result["predicted_pm25"])

                st.success(
                    f"Predicted PM2.5: {prediction:.2f} µg/m³"
                )

                st.info(f"API Latency: {latency:.2f} ms")

                prediction_category = categorize_pm25(prediction)

                st.write(
                    f"Predicted Category: **{prediction_category}**"
                )

                # Save successful prediction to a log
                log_entry = {
                    "timestamp": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    **input_data,
                    "predicted_pm25": prediction,
                    "api_latency_ms": round(latency, 2),
                    "status": "success"
                }

                os.makedirs("ml", exist_ok=True)

                pd.DataFrame([log_entry]).to_csv(
                    PREDICTION_LOG,
                    mode="a",
                    header=not os.path.exists(PREDICTION_LOG),
                    index=False
                )

        else:
            st.error(
                f"API returned status code {response.status_code}."
            )
            st.code(response.text[:1000])

    except requests.exceptions.Timeout:
        st.error("The prediction API timed out.")

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to FastAPI. Check that the API "
            "is running on port 8000."
        )

    except requests.exceptions.RequestException as e:
        st.error(f"Prediction request failed: {e}")

    except (ValueError, TypeError) as e:
        st.error(f"Invalid prediction result: {e}")

# ---------------------------------------------------------
# MLOPS MONITORING
# ---------------------------------------------------------

st.header("MLOps Monitoring")

# ---------------------------------------------------------
# MONITORING 1: MISSING DATA
# ---------------------------------------------------------

st.subheader("1. Missing Data Monitoring")

missing_values = filtered_df.isnull().sum()
missing_total = int(missing_values.sum())

if missing_total == 0:
    st.success("No missing values detected in the selected data.")
else:
    st.warning(
        f"{missing_total} missing values detected in the selected data."
    )

    st.dataframe(
        missing_values[missing_values > 0].rename(
            "Missing Values"
        ).to_frame(),
        use_container_width=True
    )

# ---------------------------------------------------------
# MONITORING 2: FEATURE DRIFT
# ---------------------------------------------------------

st.subheader("2. Feature Drift Monitoring")

numeric_features = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "precipitation",
    "wind_speed_10m"
]

drift_results = []
DRIFT_THRESHOLD = 20.0

for feature in numeric_features:
    if feature not in filtered_df.columns:
        continue

    values = pd.to_numeric(
        filtered_df[feature],
        errors="coerce"
    ).dropna()

    if len(values) < 2:
        continue

    historical_mean = values.mean()
    recent_mean = values.tail(50).mean()

    if abs(historical_mean) > 1e-9:
        drift_percent = (
            abs(recent_mean - historical_mean)
            / abs(historical_mean)
        ) * 100
    elif abs(recent_mean) <= 1e-9:
        drift_percent = 0.0
    else:
        drift_percent = None

    drift_results.append({
        "Feature": feature,
        "Historical Mean": round(historical_mean, 2),
        "Recent Mean (last 50)": round(recent_mean, 2),
        "Drift (%)": (
            round(drift_percent, 2)
            if drift_percent is not None
            else "N/A"
        ),
        "Status": (
            "High drift"
            if drift_percent is not None
            and drift_percent > DRIFT_THRESHOLD
            else "Within threshold"
            if drift_percent is not None
            else "Cannot calculate"
        )
    })

if drift_results:
    drift_df = pd.DataFrame(drift_results)

    st.dataframe(
        drift_df,
        use_container_width=True
    )

    st.caption(
        "Drift is estimated by comparing recent and historical "
        "means. The 20% threshold is a project setting."
    )
else:
    st.info("Not enough numeric feature data to calculate drift.")

# ---------------------------------------------------------
# MONITORING 3: PREDICTION LOG
# ---------------------------------------------------------

st.subheader("3. Prediction and Error Monitoring")

st.write(
    "Successful predictions are saved in a log. Actual prediction "
    "error requires actual PM2.5 values for comparison."
)

if os.path.exists(PREDICTION_LOG):
    try:
        prediction_log = pd.read_csv(PREDICTION_LOG)

        if not prediction_log.empty:
            st.dataframe(
                prediction_log.tail(10),
                use_container_width=True
            )

            if "api_latency_ms" in prediction_log.columns:
                latencies = pd.to_numeric(
                    prediction_log["api_latency_ms"],
                    errors="coerce"
                ).dropna()

                if not latencies.empty:
                    st.metric(
                        "Average Logged Prediction Latency",
                        f"{latencies.mean():.2f} ms"
                    )
        else:
            st.info("The prediction log is empty.")

    except Exception as e:
        st.info(f"Prediction log could not be read: {e}")
else:
    st.info(
        "Prediction log is not available yet. Make a successful "
        "prediction to create it."
    )

# ---------------------------------------------------------
# MONITORING 4: API STATUS
# ---------------------------------------------------------

st.subheader("4. API Monitoring")

try:
    api_start = time.time()

    api_response = requests.get(
        API_DOCS_URL,
        timeout=5
    )

    api_latency = (time.time() - api_start) * 1000

    if api_response.status_code == 200:
        st.success("FastAPI service is running.")

        st.metric(
            "API Response Time",
            f"{api_latency:.2f} ms"
        )
    else:
        st.warning(
            f"FastAPI returned status code {api_response.status_code}."
        )

except requests.exceptions.RequestException:
    st.error(
        "FastAPI service is unavailable. Check whether the API "
        "server or Docker container is running."
    )

# ---------------------------------------------------------
# MONITORING 5: PRODUCTION MODEL
# ---------------------------------------------------------

st.subheader("5. Production Model")

if os.path.exists(REGISTRY_FILE):
    try:
        with open(
            REGISTRY_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            model_info = json.load(file)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Model Version",
                str(model_info.get("model_version", "N/A"))
            )

        with col2:
            st.metric(
                "Model Type",
                str(model_info.get("model_type", "N/A"))
            )

        with col3:
            st.metric(
                "Status",
                str(model_info.get("status", "N/A"))
            )

    except Exception as e:
        st.info(
            f"Model registry information could not be loaded: {e}"
        )
else:
    st.info("Model registry file not found.")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Air Quality Index Analysis and Prediction | "
    "Data Engineering + Machine Learning + MLOps"
)

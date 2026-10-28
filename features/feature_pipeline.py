import pandas as pd


INPUT_FILE = "data/cleaned/air_quality_weather_joined.csv"
OUTPUT_FILE = "data/cleaned/air_quality_features.csv"


def create_features():
    # Load dataset
    df = pd.read_csv(INPUT_FILE)

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Lag features
    df["pm25_lag_1"] = df["value"].shift(1)
    df["pm25_lag_3"] = df["value"].shift(3)

    # Rolling features
    df["pm25_rolling_mean_3"] = (
        df["value"]
        .shift(1)
        .rolling(window=3)
        .mean()
    )

    df["pm25_rolling_mean_6"] = (
        df["value"]
        .shift(1)
        .rolling(window=6)
        .mean()
    )

    # Remove rows with missing feature values
    df = df.dropna().reset_index(drop=True)

    # Save feature dataset
    df.to_csv(OUTPUT_FILE, index=False)

    print("Feature pipeline completed successfully!")
    print(f"Rows after feature engineering: {len(df)}")
    print(f"Output file: {OUTPUT_FILE}")
    print("\nNew features:")
    print("pm25_lag_1")
    print("pm25_lag_3")
    print("pm25_rolling_mean_3")
    print("pm25_rolling_mean_6")


if __name__ == "__main__":
    create_features()
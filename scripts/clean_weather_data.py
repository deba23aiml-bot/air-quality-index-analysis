import pandas as pd
from pathlib import Path

project_folder = Path("C:/Users/deeks/air_quality_project")

input_file = (
    project_folder
    / "data"
    / "staging"
    / "new_delhi_weather.csv"
)

output_folder = project_folder / "data" / "cleaned"
output_folder.mkdir(parents=True, exist_ok=True)

output_file = output_folder / "new_delhi_weather_clean.csv"

print("Reading weather data...")

df = pd.read_csv(input_file)

print("Original rows:", len(df))

# Convert weather timestamp to New Delhi timezone
df["time"] = pd.to_datetime(df["time"], errors="coerce")
df["time"] = df["time"].dt.tz_localize("Asia/Kolkata")

# Convert weather columns to numbers
weather_columns = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "precipitation",
    "wind_speed_10m",
]

for column in weather_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove invalid timestamps
df = df.dropna(subset=["time"])

# Remove duplicate timestamps
df = df.drop_duplicates(subset=["time"])

# Sort by time
df = df.sort_values("time")

# Save cleaned weather data
df.to_csv(output_file, index=False)

print("Cleaned rows:", len(df))
print("Weather cleaning completed successfully!")
print("File:", output_file)
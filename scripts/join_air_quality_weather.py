import pandas as pd
from pathlib import Path

project_folder = Path("C:/Users/deeks/air_quality_project")

air_quality_file = (
    project_folder
    / "data"
    / "cleaned"
    / "sensor_23534_measurements_clean.csv"
)

weather_file = (
    project_folder
    / "data"
    / "cleaned"
    / "new_delhi_weather_clean.csv"
)

output_file = (
    project_folder
    / "data"
    / "cleaned"
    / "air_quality_weather_joined.csv"
)

print("Reading air-quality data...")
aq = pd.read_csv(air_quality_file)

print("Reading weather data...")
weather = pd.read_csv(weather_file)

print("Air-quality rows before join:", len(aq))
print("Weather rows before join:", len(weather))

# Convert air-quality timestamps to UTC
aq["timestamp"] = pd.to_datetime(
    aq["timestamp"],
    errors="coerce",
    utc=True
)

# Convert weather timestamps to UTC
weather["timestamp"] = pd.to_datetime(
    weather["time"],
    errors="coerce",
    utc=True
)

# Remove invalid timestamps
aq = aq.dropna(subset=["timestamp"])
weather = weather.dropna(subset=["timestamp"])

# Remove the original weather timestamp column
weather = weather.drop(columns=["time"])

print("Air-quality rows after timestamp cleaning:", len(aq))
print("Weather rows after timestamp cleaning:", len(weather))

# Join the two datasets using timestamp
joined = pd.merge(
    aq,
    weather,
    on="timestamp",
    how="inner"
)

# Remove duplicate records
joined = joined.drop_duplicates()

# Sort by timestamp
joined = joined.sort_values("timestamp")

# Save joined dataset
joined.to_csv(output_file, index=False)

print("Joined rows:", len(joined))
print("Joined columns:", len(joined.columns))
print("Weather enrichment completed successfully!")
print("File:", output_file)
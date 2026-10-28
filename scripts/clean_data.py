import pandas as pd
from pathlib import Path

# Project folder on Windows
project_folder = Path(r"C:\Users\deeks\air_quality_project")

# Input and output files
input_file = (
    project_folder
    / "data"
    / "staging"
    / "sensor_23534_measurements.csv"
)

output_folder = (
    project_folder
    / "data"
    / "cleaned"
)

# Create cleaned folder
output_folder.mkdir(parents=True, exist_ok=True)

# Read staging data
df = pd.read_csv(input_file)

print("Original rows:", len(df))

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

# Convert value to numeric
df["value"] = pd.to_numeric(
    df["value"],
    errors="coerce"
)

# Remove rows with missing important values
df = df.dropna(
    subset=[
        "timestamp",
        "pollutant",
        "unit",
        "value",
        "sensor_id"
    ]
)

# Remove duplicate rows
df = df.drop_duplicates()

# Remove negative pollutant values
df = df[df["value"] >= 0]

# Sort by timestamp
df = df.sort_values("timestamp")

# Save cleaned data
output_file = (
    output_folder
    / "sensor_23534_measurements_clean.csv"
)

df.to_csv(
    output_file,
    index=False
)

print("Cleaned rows:", len(df))
print("Rows removed:", 1000 - len(df))
print("Cleaned data saved successfully!")
print("File:", output_file)
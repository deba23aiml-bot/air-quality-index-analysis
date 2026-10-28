import json
import pandas as pd
from pathlib import Path

# Project folder
project_folder = Path(r"C:\Users\deeks\air_quality_project")

# File locations
input_file = project_folder / "data" / "raw" / "sensor_23534_measurements.json"
output_folder = project_folder / "data" / "staging"

# Create staging folder
output_folder.mkdir(parents=True, exist_ok=True)

# Read raw JSON
with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)

# Get measurements
measurements = data["results"]

# Convert to a simple table
rows = []

for measurement in measurements:

    rows.append({
        "timestamp": measurement["period"]["datetimeFrom"]["local"],
        "pollutant": measurement["parameter"]["name"],
        "unit": measurement["parameter"]["units"],
        "value": measurement["value"],
        "sensor_id": 23534
    })

# Create DataFrame
df = pd.DataFrame(rows)

# Save staging CSV
output_file = output_folder / "sensor_23534_measurements.csv"

df.to_csv(output_file, index=False)

print("Conversion successful!")
print("Rows:", len(df))
print("Columns:", list(df.columns))
print("File:", output_file)
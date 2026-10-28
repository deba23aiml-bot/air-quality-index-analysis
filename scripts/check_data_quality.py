
import pandas as pd
from pathlib import Path

# Project folder
project_folder = Path(r"C:\Users\deeks\air_quality_project")

# Read staging CSV
input_file = (
    project_folder
    / "data"
    / "staging"
    / "sensor_23534_measurements.csv"
)

df = pd.read_csv(input_file)

print("===== DATA QUALITY CHECK =====")

# 1. Number of rows and columns
print("\n1. Dataset size:")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# 2. Column names
print("\n2. Columns:")
print(list(df.columns))

# 3. Missing values
print("\n3. Missing values:")
print(df.isnull().sum())

# 4. Duplicate rows
print("\n4. Duplicate rows:")
print(df.duplicated().sum())

# 5. Pollutants
print("\n5. Pollutants:")
print(df["pollutant"].value_counts())

# 6. Units
print("\n6. Units:")
print(df["unit"].value_counts())

# 7. Value statistics
print("\n7. Value statistics:")
print(df["value"].describe())

# 8. Timestamp check
df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

print("\n8. Invalid timestamps:")
print(df["timestamp"].isnull().sum())

print("\n===== CHECK COMPLETE =====")
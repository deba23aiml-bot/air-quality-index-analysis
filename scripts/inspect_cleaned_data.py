import pandas as pd
from pathlib import Path

project_folder = Path(r"C:\Users\deeks\air_quality_project")

input_file = (
    project_folder
    / "data"
    / "cleaned"
    / "sensor_23534_measurements_clean.csv"
)

df = pd.read_csv(input_file)

print("===== CLEANED DATASET =====")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nNumber of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nPollutant:")
print(df["pollutant"].unique())

print("\nUnits:")
print(df["unit"].unique())

print("\nValue range:")
print("Minimum:", df["value"].min())
print("Maximum:", df["value"].max())

print("\nTime range:")
print("From:", df["timestamp"].min())
print("To:", df["timestamp"].max())

print("\n===== INSPECTION COMPLETE =====")
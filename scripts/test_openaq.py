import requests
from pathlib import Path
from dotenv import dotenv_values
import json

# Project folder
project_folder = Path(r"C:\Users\deeks\air_quality_project")

# Read API key from .env
config = dotenv_values(project_folder / ".env")
api_key = config.get("OPENAQ_API_KEY")

# Sensor we are collecting
sensor_id = 23534

# OpenAQ API URL
url = f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements"

# API key
headers = {
    "X-API-Key": api_key
}

# Get 1,000 measurements
params = {
    "limit": 1000
}

# Send request
response = requests.get(
    url,
    headers=headers,
    params=params
)

print("Status code:", response.status_code)

if response.status_code == 200:

    data = response.json()

    print("Number of measurements:", len(data["results"]))

    # Raw data folder
    raw_folder = project_folder / "data" / "raw"
    raw_folder.mkdir(parents=True, exist_ok=True)

    # File name
    output_file = raw_folder / "sensor_23534_measurements.json"

    # Save the complete API response
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    print("Raw data saved successfully!")
    print("File:", output_file)

else:

    print("Error:")
    print(response.text)
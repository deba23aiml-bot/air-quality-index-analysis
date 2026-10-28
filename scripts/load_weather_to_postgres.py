import pandas as pd
import psycopg2

project_folder = "C:/Users/deeks/air_quality_project"

csv_file = (
    project_folder
    + "/data/cleaned/air_quality_weather_joined.csv"
)

print("Reading joined CSV...")

df = pd.read_csv(csv_file)

print("Rows to load:", len(df))

# PostgreSQL connection
connection = psycopg2.connect(
    host="localhost",
    port=5432,
    database="AQI_Project",
    user="postgres",
    password=input("Enter PostgreSQL password: ")
)

cursor = connection.cursor()

print("Connected to PostgreSQL.")

for _, row in df.iterrows():

    cursor.execute(
        """
        INSERT INTO air_quality_weather
        (
            timestamp,
            pollutant,
            unit,
            value,
            sensor_id,
            temperature_2m,
            relative_humidity_2m,
            pressure_msl,
            precipitation,
            wind_speed_10m
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            row["timestamp"],
            row["pollutant"],
            row["unit"],
            row["value"],
            row["sensor_id"],
            row["temperature_2m"],
            row["relative_humidity_2m"],
            row["pressure_msl"],
            row["precipitation"],
            row["wind_speed_10m"],
        )
    )

connection.commit()

cursor.close()
connection.close()

print("Data loaded successfully!")
print("Rows inserted:", len(df))
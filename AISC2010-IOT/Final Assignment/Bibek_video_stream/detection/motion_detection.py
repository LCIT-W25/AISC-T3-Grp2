import requests
import csv
import os
import datetime
import time
import threading
from config import PHYPOX_URL

CSV_FILE = "sensor_data.csv"

# Ensure CSV file exists and has headers
def initialize_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ", "latitude", "longitude", "altitude"])

# Function to fetch and store data from Phyphox API
def fetch_phyphox_data():
    while True:
        try:
            response = requests.get(PHYPOX_URL)
            response.raise_for_status()
            data = response.json()

            timestamp = datetime.datetime.now().isoformat()
            accX = data.get("buffer", {}).get("accX", {}).get("buffer", [None])[-1]
            accY = data.get("buffer", {}).get("accY", {}).get("buffer", [None])[-1]
            accZ = data.get("buffer", {}).get("accZ", {}).get("buffer", [None])[-1]
            gyroX = data.get("buffer", {}).get("gyrX", {}).get("buffer", [None])[-1]
            gyroY = data.get("buffer", {}).get("gyrY", {}).get("buffer", [None])[-1]
            gyroZ = data.get("buffer", {}).get("gyrZ", {}).get("buffer", [None])[-1]
            latitude = data.get("buffer", {}).get("locLat", {}).get("buffer", [None])[-1]
            longitude = data.get("buffer", {}).get("locLon", {}).get("buffer", [None])[-1]
            altitude = data.get("buffer", {}).get("locZ", {}).get("buffer", [None])[-1]

            with open(CSV_FILE, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, accX, accY, accZ, gyroX, gyroY, gyroZ, latitude, longitude, altitude])

            print(f"📡 Fetched Phyphox Data → Acc({accX}, {accY}, {accZ}) Gyro({gyroX}, {gyroY}, {gyroZ}) Loc({latitude}, {longitude}, {altitude})")

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Could not fetch from Phyphox: {e}")

        time.sleep(1)

if __name__ == "__main__":
    initialize_csv()
    fetch_phyphox_data()
from flask import Flask, request
from gtts import gTTS
import os
import cv2
import numpy as np
import requests
import threading
import time

app = Flask(__name__)

# Replace with your phone's IP address
CAMERA_URL = 'http://10.0.0.218:8080/shot.jpg'  # e.g., http://192.168.1.101:8080/shot.jpg

# Global variables to store real-time sensor data
sensor_data = {
    'acc_x': None,
    'gyro_z': None,
    'sound_level': None
}

# Function to continuously get camera frames
def get_camera_frame():
    try:
        response = requests.get(CAMERA_URL)
        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(img_array, -1)
        if frame is not None:
            # Resize the frame to 160x320
            frame = cv2.resize(frame, (320, 160))
        return frame
    except Exception as e:
        print("Camera fetch error:", e)
        return None

# Function to detect lane or object in the camera frame
def detect_lane_or_object(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Simple logic (in real use: object detection model or line detection)
    edge_pixels = np.sum(edges > 0)
    print("Edge Pixel Count:", edge_pixels)

    if edge_pixels > 50000:
        return "Obstacle or complex scene detected"
    else:
        return "Road clear"

# Function to simulate continuously receiving sensor data from the phone
def receive_sensor_data_continuously():
    while True:
        # Fetch the latest sensor data (real data from the phone should come here)
        global sensor_data
        if sensor_data['acc_x'] is None or sensor_data['gyro_z'] is None or sensor_data['sound_level'] is None:
            print("Sensor data: None")
        else:
            print("Sensor Data:", sensor_data)

        # Process sensor data and give instruction
        instruction = process_sensor_data(sensor_data)
        print("Instruction:", instruction)

        # Wait for the next iteration (e.g., 1 second)
        time.sleep(1)

# Function to process the sensor data and generate an instruction
def process_sensor_data(data):
    acc_x = data['acc_x']
    gyro_z = data['gyro_z']
    sound_level = data['sound_level']

    instruction = "Keep driving"

    # Camera input check
    frame = get_camera_frame()
    if frame is not None:
        camera_feedback = detect_lane_or_object(frame)
        print("Camera feedback:", camera_feedback)
        if "Obstacle" in camera_feedback:
            instruction = "Obstacle ahead, slow down"

    # Sensor logic for instructions
    if acc_x is not None and acc_x > 2:
        instruction = "Accelerating"
    elif gyro_z is not None and gyro_z > 3:
        instruction = "Turning left"
    elif sound_level is not None and sound_level > 85:
        instruction = "Siren or horn detected"

    # Text to speech (convert to spoken instruction)
    tts = gTTS(instruction)
    tts.save("instruction.mp3")
    os.system("mpg321 instruction.mp3")

    return instruction

@app.route('/sensor', methods=['POST'])
def receive_sensor_data():
    global sensor_data
    data = request.json
    print("Received Data:", data)

    # Update global sensor data with incoming data
    sensor_data['acc_x'] = data.get("acc_x", sensor_data['acc_x'])
    sensor_data['gyro_z'] = data.get("gyro_z", sensor_data['gyro_z'])
    sensor_data['sound_level'] = data.get("sound_level", sensor_data['sound_level'])

    instruction = process_sensor_data(sensor_data)

    return {"status": "OK", "instruction": instruction}

if __name__ == '__main__':
    # Start the background thread to simulate continuous sensor data reception
    threading.Thread(target=receive_sensor_data_continuously, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
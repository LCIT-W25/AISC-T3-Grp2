from flask import Flask, request, Response
import os
import cv2
import numpy as np
import requests
import threading
import time

app = Flask(__name__)

# IP address of your phone (DroidCam or IP Webcam app)
# Update this if your phone IP changes
CAMERA_URL = "http://192.168.2.11:8080/shot.jpg"

# Global sensor state (optional if still used)
sensor_data = {
    'acc_x': None,
    'gyro_z': None,
    'sound_level': None
}

# Function to get a frame from the phone camera


def get_camera_frame():
    try:
        response = requests.get(CAMERA_URL, timeout=2)
        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(img_array, -1)
        if frame is not None:
            frame = cv2.resize(frame, (320, 160))
        return frame
    except Exception as e:
        print("[APP] Camera fetch error:", e)
        return None

# ✅ Endpoint to serve camera image to UdacitySimEnv


@app.route('/camera', methods=['GET'])
def get_camera_image():
    frame = get_camera_frame()
    if frame is not None:
        _, jpeg = cv2.imencode('.jpg', frame)
        return Response(jpeg.tobytes(), mimetype='image/jpeg')
    return "No camera image", 500

# ✅ Endpoint to receive sensor data (if still used)


@app.route('/sensor', methods=['POST'])
def receive_sensor_data():
    global sensor_data
    data = request.json
    print("[APP] Received Sensor Data:", data)

    # Update global sensor data
    sensor_data['acc_x'] = data.get("acc_x", sensor_data['acc_x'])
    sensor_data['gyro_z'] = data.get("gyro_z", sensor_data['gyro_z'])
    sensor_data['sound_level'] = data.get(
        "sound_level", sensor_data['sound_level'])

    return {"status": "OK", "message": "Sensor data received"}

# Optional: background loop for debug logging


def receive_sensor_data_continuously():
    while True:
        if None not in sensor_data.values():
            print("[APP] Sensor State:", sensor_data)
        time.sleep(1)


if __name__ == '__main__':
    # Start background logging thread (optional)
    threading.Thread(target=receive_sensor_data_continuously,
                     daemon=True).start()
    app.run(host='0.0.0.0', port=5000)

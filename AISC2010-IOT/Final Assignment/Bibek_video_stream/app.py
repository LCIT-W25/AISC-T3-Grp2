from flask import Flask, jsonify, render_template, Response, send_file
from flask_sock import Sock
from detection.ipwebcam import get_video_frame
# from detection.lane_detector
from utils.logger import log_event
import threading
import time
import json
from config import IP_WEBCAM_HOST, IP_WEBCAM_PORT, EVENT_MESSAGES
import requests

from detection.motion_detection import fetch_phyphox_data, initialize_csv

app = Flask(__name__)
sock = Sock(app)
clients = []

running = True

def continuous_detection_loop():
    print("🌀 Continuous detection loop started...")
    while running:
        print("🔁 Loop tick")
        frame = get_video_frame()
        if frame is not None:
            print("📸 Frame captured successfully.")
        else:
            print("⛔ No frame captured. Skipping visual detection.")

        time.sleep(5)

@app.route("/")
def dashboard():
    return render_template("dashboard.html", ip=IP_WEBCAM_HOST, port=IP_WEBCAM_PORT)

if __name__ == "__main__":
    initialize_csv()
    threading.Thread(target=fetch_phyphox_data, daemon=True).start()

    detection_thread = threading.Thread(target=continuous_detection_loop, daemon=True)
    detection_thread.start()
    app.run(port=5000, debug=False, use_reloader=False)

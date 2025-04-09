import socketio
import eventlet
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import threading


class RawSimulatorBridge:
    def __init__(self):
        self.sio = socketio.Server()
        self.app = socketio.WSGIApp(self.sio)
        self.connected = False
        self.last_image = None
        self.last_telemetry = None
        self.current_action = {'steering_angle': "0.0",
                               'throttle': "0.0", 'brake': "0.0"}
        self.reset = False

        @self.sio.event
        def connect(sid, environ):
            print(f"[SIM] Connected with session id {sid}")
            self.connected = True

        @self.sio.on('telemetry')
        def telemetry(sid, data):
            img_str = data["image"]
            image = Image.open(BytesIO(base64.b64decode(img_str)))
            image_array = np.asarray(image)

            self.last_image = image_array
            self.last_telemetry = {
                'steering_angle': float(data["steering_angle"]),
                'throttle': float(data["throttle"]),
                'speed': float(data["speed"])
            }

            self.sio.emit("steer", data=self.current_action, to=sid)
            if self.reset:
                self.sio.emit("reset", data=self.current_action, to=sid)
                self.reset = False  # only trigger once

        @self.sio.on('manual')
        def manual(sid, data):
            print("[SIM] Manual control (ignored)")

    def set_action(self, steer, throttle, brake=0.0):
        self.current_action = {
            'steering_angle': str(steer),
            'throttle': str(throttle),
            'brake': str(brake)
        }

    def run(self):
        print("[SERVER] Raw SocketIO server on port 4567...")
        eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 4567)), self.app)

    def start_async(self):
        thread = threading.Thread(target=self.run)
        thread.daemon = True
        thread.start()

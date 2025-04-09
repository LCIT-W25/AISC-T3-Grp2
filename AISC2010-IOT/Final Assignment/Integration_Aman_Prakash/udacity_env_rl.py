# udacity_env.py
import gym
from gym import spaces
import numpy as np
import time
import requests
import cv2
from sim_bridge import RawSimulatorBridge as SimulatorBridge


class UdacitySimEnv(gym.Env):
    def __init__(self, predict=False):
        super(UdacitySimEnv, self).__init__()

        self.image_shape = (160, 320, 3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        self.action_space = spaces.Box(low=np.array(
            [-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        self.predict = predict
        self.last_obs = None
        self.step_delay = 0.1

        self.off_track_count = 0
        self.speed_below_threshold_count = 0

        if not self.predict:
            self.bridge = SimulatorBridge()
            self.bridge.start_async()

    def fetch_image_from_flask(self):
        try:
            response = requests.get("http://localhost:5000/camera", timeout=2)
            img_array = np.array(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (320, 160))
            return image
        except Exception as e:
            print("[PREDICT] Camera fetch failed:", e)
            return np.zeros(self.image_shape, dtype=np.uint8)

    def reset(self):
        if self.predict:
            image = self.fetch_image_from_flask()
            self.last_obs = image
            return image
        else:
            print("[ENV] Reset triggered")
            self.bridge.reset = True
            self.bridge.set_action(0.0, 0.0, 1.0)
            time.sleep(1.0)

            for _ in range(50):
                if self.bridge.last_image is not None:
                    break
                time.sleep(0.1)

            if self.bridge.last_image is None:
                print("[ENV] Failed to get image after reset.")
                return np.zeros(self.image_shape, dtype=np.uint8)

            self.off_track_count = 0
            self.speed_below_threshold_count = 0
            self.last_obs = self.bridge.last_image
            return self.last_obs

    def step(self, action):
        steer = float(action[0])
        throttle = float(action[1])

        if self.predict:
            image = self.fetch_image_from_flask()
            self.last_obs = image

            # Send action to audio server
            try:
                requests.post("http://localhost:5005/say", json={
                    "steer": steer,
                    "throttle": throttle
                })
            except Exception as e:
                print("[AUDIO ERROR]", e)

            return image, 0.0, False, {}
        else:
            self.bridge.reset = False
            self.bridge.set_action(steer, throttle, 0.0)
            time.sleep(self.step_delay)

            obs = self.bridge.last_image
            telemetry = self.bridge.last_telemetry

            if obs is None or telemetry is None:
                return self.last_obs, -10.0, True, {}

            speed = telemetry["speed"]

            if speed < 1.0:
                self.speed_below_threshold_count += 1
            else:
                self.speed_below_threshold_count = 0

            is_off = self.is_off_track(obs)
            if is_off:
                self.off_track_count += 1
            else:
                self.off_track_count = 0

            reward = self.compute_reward(speed, is_off)
            done = self.check_done()

            self.last_obs = obs
            return obs, reward, done, {}

    def compute_reward(self, speed, is_off):
        reward = 0.0
        if speed < 1.0:
            reward -= 1.0
        else:
            reward += speed / 10.0

        if is_off:
            reward -= 5.0

        return reward

    def check_done(self):
        return self.speed_below_threshold_count > 30 or self.off_track_count > 20

    def is_off_track(self, image_array):
        h, w, _ = image_array.shape
        bottom = image_array[int(h * 0.66):, :, :]

        red = (bottom[:, :, 0] > 180) & (
            bottom[:, :, 1] < 80) & (bottom[:, :, 2] < 80)
        white = (bottom[:, :, 0] > 200) & (
            bottom[:, :, 1] > 200) & (bottom[:, :, 2] > 200)

        curb = red | white
        curb_ratio = np.sum(curb) / curb.size

        greenish = (bottom[:, :, 1] > 100) & (
            bottom[:, :, 0] < 100) & (bottom[:, :, 2] < 100)
        brownish = (bottom[:, :, 0] > 100) & (
            bottom[:, :, 1] > 100) & (bottom[:, :, 2] < 100)
        dirt = greenish | brownish
        dirt_ratio = np.sum(dirt) / dirt.size

        return curb_ratio > 0.05 or dirt_ratio > 0.25

    def render(self, mode="human"):
        pass

    def close(self):
        pass

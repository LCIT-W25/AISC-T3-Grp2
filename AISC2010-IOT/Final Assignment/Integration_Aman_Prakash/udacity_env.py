import gym
from gym import spaces
import numpy as np
import time
import requests  # ✅ added for Flask audio call
from sim_bridge import RawSimulatorBridge as SimulatorBridge

class UdacitySimEnv(gym.Env):
    def __init__(self):
        super(UdacitySimEnv, self).__init__()

        self.bridge = SimulatorBridge()
        self.bridge.start_async()

        print("[ENV] Waiting for simulator to connect...")
        while not self.bridge.connected:
            time.sleep(0.5)
        print("[ENV] Simulator connected!")

        self.image_shape = (160, 320, 3)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8),
            "state": spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32),
        })
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        self.last_obs = None
        self.step_delay = 0.1

        self.last_action = np.array([0.0, 0.0])
        self.off_track_count = 0
        self.speed_below_threshold_count = 0
        self.speed_history = []

        self.step_counter = 0  # ✅ counter to limit audio frequency

    def reset(self):
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
            return {
                "image": np.zeros(self.image_shape, dtype=np.uint8),
                "state": np.array([0.0, 0.0], dtype=np.float32)
            }

        self.off_track_count = 0
        self.speed_below_threshold_count = 0
        self.speed_history = []
        self.last_action = np.array([0.0, 0.0])
        self.last_obs = self.bridge.last_image
        self.step_counter = 0  # ✅ reset step counter
        return {
            "image": self.bridge.last_image,
            "state": self.last_action.copy()
        }

    def step(self, action):
        steer = float(action[0])
        throttle = float(action[1])

        self.bridge.reset = False
        self.bridge.set_action(steer, throttle, 0.0)
        time.sleep(self.step_delay)

        self.step_counter += 1  # ✅ increment step counter
        if self.step_counter % 10 == 0:  # approx every 1s if step_delay = 0.1
            try:
                requests.post("http://localhost:5005/say", json={
                    "steer": steer,
                    "throttle": throttle
                })
            except Exception as e:
                print(f"[AUDIO ERROR] {e}")

        obs = self.bridge.last_image
        telemetry = self.bridge.last_telemetry

        if obs is None or telemetry is None:
            return self.last_obs, -10.0, True, {}

        speed = telemetry["speed"]
        self.speed_history.append(speed)
        if len(self.speed_history) > 10:
            self.speed_history.pop(0)

        if speed < 5.0:
            self.speed_below_threshold_count += 1
        else:
            self.speed_below_threshold_count = 0

        is_off = self.is_off_track(obs)
        if is_off:
            self.off_track_count += 1
        else:
            self.off_track_count = 0

        reward = self.compute_reward(speed, is_off, action)
        done = self.check_done()

        self.last_action = np.array(action)
        self.last_obs = {
            "image": obs,
            "state": self.last_action.copy()
        }

        return self.last_obs, reward, done, {}

    def compute_reward(self, speed, is_off, action):
        reward = 0.0

        if is_off:
            reward -= 5.0
        else:
            reward += 1

        if speed < 10.0:
            reward -= 2.0
        elif speed < 15.0:
            reward += (speed - 10.0) * 0.2
        elif speed <= 25.0:
            reward += 1.0
        else:
            reward += 1.5

        if len(self.speed_history) >= 5:
            speed_std = np.std(self.speed_history)
            reward += max(0, 0.5 - speed_std)

        steer_delta = abs(action[0] - self.last_action[0])
        throttle_delta = abs(action[1] - self.last_action[1])

        reward -= 2.0 * steer_delta
        reward -= 1.5 * throttle_delta

        return reward

    def check_done(self):
        if self.speed_below_threshold_count > 30:
            return True
        if self.off_track_count > 20:
            return True
        return False

    def is_off_track(self, image_array):
        h, w, _ = image_array.shape
        bottom = image_array[int(h * 0.66):, :, :]

        red = (bottom[:, :, 0] > 180) & (bottom[:, :, 1] < 80) & (bottom[:, :, 2] < 80)
        white = (bottom[:, :, 0] > 200) & (bottom[:, :, 1] > 200) & (bottom[:, :, 2] > 200)

        curb = red | white
        curb_ratio = np.sum(curb) / curb.size

        greenish = (bottom[:, :, 1] > 100) & (bottom[:, :, 0] < 100) & (bottom[:, :, 2] < 100)
        brownish = (bottom[:, :, 0] > 100) & (bottom[:, :, 1] > 100) & (bottom[:, :, 2] < 100)
        dirt = greenish | brownish
        dirt_ratio = np.sum(dirt) / dirt.size

        return curb_ratio > 0.05 or dirt_ratio > 0.25

    def render(self, mode="human"):
        pass

    def close(self):
        pass
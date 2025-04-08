from stable_baselines3 import PPO
import numpy as np
import requests
import time

# Load model
model = PPO.load("ppo-udacity_200000_steps")

# Dummy observation: (3, 160, 320)
dummy_image = np.zeros((3, 160, 320), dtype=np.uint8)

# Loop forever, sending a new action every 1 second
while True:
    action, _ = model.predict(dummy_image)
    steer, throttle = float(action[0]), float(action[1])

    print(f"[MODEL] Raw output: steer={steer:.3f}, throttle={throttle:.3f}")

    try:
        response = requests.post("http://localhost:5005/say", json={
            "steer": steer,
            "throttle": throttle
        })
        print(f"[AUDIO] {response.json()['status']}")
    except Exception as e:
        print(f"[ERROR] Failed to send to audio server: {e}")

    time.sleep(1)  # wait 1 second before next action

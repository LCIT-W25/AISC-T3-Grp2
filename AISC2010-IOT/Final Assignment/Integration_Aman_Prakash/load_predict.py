from stable_baselines3 import PPO
import numpy as np
import requests

# Load PPO model
model = PPO.load("ppo-udacity_200000_steps")  # Make sure this is the correct model folder or file

# Create dummy observation matching your environment: (3, 160, 320)
dummy_image = np.zeros((3, 160, 320), dtype=np.uint8)

# Predict action
action, _ = model.predict(dummy_image)
steer, throttle = float(action[0]), float(action[1])

# Output
print(f"[MODEL] Raw output: steer={steer:.3f}, throttle={throttle:.3f}")

# Send to Flask audio server
try:
    response = requests.post("http://localhost:5005/say", json={
        "steer": steer,
        "throttle": throttle
    })
    print(f"[MODEL] Audio server responded: {response.json()}")
except Exception as e:
    print(f"[MODEL] Error sending to audio server: {e}")
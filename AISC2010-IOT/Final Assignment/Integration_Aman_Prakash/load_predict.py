# load_predict.py
from stable_baselines3 import PPO
from udacity_env_rl import UdacitySimEnv
import numpy as np
import datetime
import requests
import time

# Load model
print("[MODEL] Loading PPO model...")
model = PPO.load("ppo-udacity_200000_steps")

# Logging setup
log_file = open("inference_log.csv", "a")
log_file.write("timestamp,steering,throttle,speech_output\n")


def format_speech(steer, throttle):
    steer_deg = abs(steer * 25)
    direction = "Right" if steer > 0 else "Left" if steer < 0 else "Straight"
    speed = throttle * 30
    motion = "Accelerate" if throttle > 0.1 else "Brake"
    if direction == "Straight":
        return f"{motion} {speed:.1f} kilometers per hour"
    else:
        return f"{direction} {steer_deg:.1f} degrees, {motion} {speed:.1f} kilometers per hour"


def run_prediction_loop():
    env = UdacitySimEnv(predict=True)
    obs = env.reset()

    while True:
        action, _ = model.predict(obs)
        steer, throttle = float(action[0]), float(action[1])
        speech = format_speech(steer, throttle)

        # Send to audio server
        try:
            requests.post("http://localhost:5005/say", json={
                "steer": steer,
                "throttle": throttle
            })
        except Exception as e:
            print("[Audio Server Error]", e)

        # Log to CSV
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(
            f"{timestamp},{steer:.3f},{throttle:.3f},\"{speech}\"\n")
        log_file.flush()

        print(
            f"[ACTION] Steer: {steer:.2f}, Throttle: {throttle:.2f} | {speech}")

        obs, _, _, _ = env.step(action)
        time.sleep(0.5)


if __name__ == "__main__":
    run_prediction_loop()

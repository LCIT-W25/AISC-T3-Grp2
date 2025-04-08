from stable_baselines3 import PPO
from udacity_env import UdacitySimEnv
import datetime

# Load the environment and model
env = UdacitySimEnv()
model = PPO.load("ppo-udacity-v2", env=env)

# Logging setup
log_file = open("inference_log.csv", "w")
log_file.write("timestamp,steering,throttle,reward,speech_output,done\n")

def format_speech(steer, throttle):
    steer_deg = abs(steer * 25)
    direction = "Right" if steer > 0 else "Left" if steer < 0 else "Straight"
    speed = throttle * 30
    motion = "Accelerate" if throttle > 0.1 else "Brake"
    if direction == "Straight":
        return f"{motion} {speed:.1f} kilometers per hour"
    else:
        return f"{direction} {steer_deg:.1f} degrees, {motion} {speed:.1f} kilometers per hour"

# Start inference
obs = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    steer, throttle = float(action[0]), float(action[1])
    speech = format_speech(steer, throttle)

    obs, reward, done, info = env.step(action)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"{timestamp},{steer:.3f},{throttle:.3f},{reward:.2f},\"{speech}\",{done}\n"
    log_file.write(log_line)
    print(log_line.strip())

log_file.close()
env.close()
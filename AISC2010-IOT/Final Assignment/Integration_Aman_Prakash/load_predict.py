from stable_baselines3 import PPO
from udacity_env import UdacitySimEnv

# Load your environment
env = UdacitySimEnv()

# Load the trained model
model = PPO.load("ppo-udacity_200000_steps", env=env)

# Run inference loop
obs = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)

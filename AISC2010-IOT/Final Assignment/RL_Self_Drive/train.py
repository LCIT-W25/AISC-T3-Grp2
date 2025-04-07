from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from udacity_env import UdacitySimEnv

# Create environment
env = UdacitySimEnv()

# Optional: Save checkpoints every 100k steps
checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path="./checkpoints/",
    name_prefix="ppo-udacity-v2"
)

# Create PPO model with custom config
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    learning_rate=2.5e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
)

# Train for 500k steps
model.learn(total_timesteps=500_000, callback=checkpoint_callback)

# Save final model
model.save("ppo-udacity-v2")

env.close()

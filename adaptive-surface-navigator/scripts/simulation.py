import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from numpngw import write_apng
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

env = gym.make("FetchReach-v3", render_mode="human")
observation, info = env.reset()

images = [env.render()]
for _ in range(200000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()

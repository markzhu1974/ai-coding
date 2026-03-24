#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from envs.ur5_pickplace_env import UR5PickPlaceEnv
import numpy as np

print("Creating env...")
env = UR5PickPlaceEnv(render_mode='rgb_array', use_gui=False, max_steps=10)
print("Env created, resetting...")
obs, info = env.reset()
print(f"Reset done, obs shape {obs.shape}")
print("Taking random action...")
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f"Step done, reward {reward}, terminated {terminated}")
env.close()
print("Env closed.")
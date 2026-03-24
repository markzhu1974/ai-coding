#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from envs.ur5_pickplace_env import UR5PickPlaceEnv
import numpy as np

env = UR5PickPlaceEnv(render_mode='rgb_array', use_gui=False, max_steps=50)
print("Testing drop condition...")

obs, info = env.reset()
print(f"Initial obs shape: {obs.shape}")
print(f"Info: {info}")

# Take random actions for a few steps
for i in range(10):
    action = env.action_space.sample()
    # Force gripper to close then open
    if i < 5:
        action[3] = 0.0  # close
    else:
        action[3] = 1.0  # open
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i}: reward {reward:.3f}, terminated {terminated}, truncated {truncated}, grabbed {info['object_grabbed']}, dropped {info.get('object_dropped', False)}")
    if terminated:
        print("Episode terminated!")
        break

env.close()
print("Test done.")
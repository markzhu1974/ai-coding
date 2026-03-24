#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from envs.ur5_pickplace_env import UR5PickPlaceEnv
import numpy as np

env = UR5PickPlaceEnv(render_mode='rgb_array', use_gui=False, max_steps=50)
print("Testing simplified grasp conditions...")

for episode in range(3):
    obs, info = env.reset()
    print(f"\nEpisode {episode}")
    print(f"Object position: {info['object_position']}")
    print(f"Target position: {info['target_position']}")
    
    total_reward = 0
    grabbed = False
    placed = False
    
    # 尝试简单的动作：向下移动并闭合夹爪
    for step in range(20):
        # 简单策略：向物体移动并闭合夹爪
        if step < 10:
            # 向物体方向移动
            action = np.array([0.0, 0.02, 0.0, 0.0])  # 轻微向物体移动，夹爪闭合
        else:
            # 夹爪保持闭合，向目标移动
            action = np.array([0.0, 0.02, 0.0, 0.0])
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if info['object_grabbed'] and not grabbed:
            grabbed = True
            print(f"  Step {step}: Object GRABBED! Reward: {reward}")
        if info.get('object_placed', False) and not placed:
            placed = True
            print(f"  Step {step}: Object PLACED! Reward: {reward}")
        if terminated:
            print(f"  Step {step}: Episode terminated")
            break
    
    print(f"Total reward: {total_reward:.2f}, Grabbed: {grabbed}, Placed: {placed}")

env.close()
print("\nTest completed.")
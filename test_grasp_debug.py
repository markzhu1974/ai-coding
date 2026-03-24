#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from envs.ur5_pickplace_env import UR5PickPlaceEnv
import numpy as np

env = UR5PickPlaceEnv(render_mode='rgb_array', use_gui=False, max_steps=50)
print("Testing grasp detection with debugging...")

obs, info = env.reset()
print(f"Object position: {info['object_position']}")
print(f"Target position: {info['target_position']}")

# 动作序列：保持夹爪闭合，轻微移动
for step in range(30):
    # 始终闭合夹爪
    action = np.array([0.0, 0.0, 0.0, 0.0])  # 无移动，夹爪完全闭合
    obs, reward, terminated, truncated, info = env.step(action)
    
    ee_pos = env.robot.get_ee_position()
    gripper = env.robot.get_gripper_state()
    print(f"Step {step}: EE={ee_pos}, gripper={gripper:.3f}, reward={reward:.3f}, grabbed={info['object_grabbed']}, dropped={info.get('object_dropped', False)}")
    
    if info['object_grabbed']:
        print("  *** OBJECT GRABBED! ***")
        # 检查物体位置
        if env.object_id is not None:
            obj_pos, _ = env.robot.p.getBasePositionAndOrientation(env.object_id, physicsClientId=env.physics_client_id)
            print(f"  Object position: {obj_pos}")
    
    if terminated:
        print("Episode terminated")
        break

env.close()
print("Test done.")
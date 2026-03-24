import sys
sys.path.insert(0, '.')
from envs.ur5_pickplace_env import UR5PickPlaceEnv

# 创建环境（无GUI）
env = UR5PickPlaceEnv(render_mode='rgb_array', use_gui=False, max_steps=50)

print(f"状态空间维度: {env.observation_space.shape}")
print(f"动作空间维度: {env.action_space.shape}")

# 测试重置
obs, info = env.reset()
print(f"观察状态形状: {obs.shape}")
print(f"预期: 11维")

# 测试随机动作
action = env.action_space.sample()
print(f"采样动作: {action}")

obs, reward, terminated, truncated, info = env.step(action)
print(f"执行后状态形状: {obs.shape}")
print(f"奖励: {reward}")
print(f"终止: {terminated}, 截断: {truncated}")

env.close()
print("测试通过!")

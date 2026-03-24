#!/usr/bin/env python3
"""
简化测试脚本，验证环境是否正常工作
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("="*60)
print("简化测试 - UR5机器人上下料环境")
print("="*60)

# 测试环境
print("\n1. 测试环境创建和基本操作...")
try:
    from envs.ur5_pickplace_env import UR5PickPlaceEnv
    
    # 创建环境（无GUI，更快）
    env = UR5PickPlaceEnv(render_mode='rgb_array', use_gui=False, max_steps=20)
    print("  ✓ 环境创建成功")
    
    # 重置环境
    obs, info = env.reset()
    print(f"  ✓ 环境重置成功")
    print(f"     状态形状: {obs.shape}")
    print(f"     物体位置: {info.get('object_position', 'N/A')}")
    
    # 测试几步随机动作
    print("  ✓ 测试随机动作...")
    total_reward = 0
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        print(f"     步骤 {i+1}: 奖励={reward:.3f}, 累计奖励={total_reward:.3f}")
        
        if done or truncated:
            print(f"     Episode 在 {i+1} 步结束")
            break
    
    env.close()
    print("  ✓ 环境测试完成")
    
except Exception as e:
    print(f"  ✗ 环境测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试模型创建
print("\n2. 测试模型创建...")
try:
    import torch
    import torch.nn as nn
    from stable_baselines3 import PPO
    
    print("  ✓ 依赖导入成功")
    
    # 创建简单的测试环境
    from envs.ur5_pickplace_env import UR5PickPlaceEnv
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    
    def make_test_env():
        env = UR5PickPlaceEnv(render_mode='rgb_array', use_gui=False, max_steps=10)
        return Monitor(env)
    
    # 创建向量化环境
    env = make_vec_env(make_test_env, n_envs=1, seed=42)
    print("  ✓ 向量化环境创建成功")
    
    # 创建模型
    policy_kwargs = {
        'net_arch': dict(pi=[32, 32], vf=[32, 32]),
        'activation_fn': nn.Tanh
    }
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        gamma=0.99,
        policy_kwargs=policy_kwargs,
        verbose=0
    )
    
    print(f"  ✓ PPO模型创建成功")
    print(f"     参数数量: {sum(p.numel() for p in model.policy.parameters()):,}")
    
    # 测试一步训练
    print("  ✓ 测试一步训练...")
    model.learn(total_timesteps=32)
    print("  ✓ 训练完成")
    
    env.close()
    
except Exception as e:
    print(f"  ✗ 模型测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试配置文件
print("\n3. 测试配置文件...")
try:
    import yaml
    
    # 读取环境配置
    with open('configs/env_config.yaml', 'r') as f:
        env_config = yaml.safe_load(f)
    print(f"  ✓ 环境配置加载成功")
    print(f"     最大步数: {env_config['env']['max_steps']}")
    
    # 读取训练配置
    with open('configs/train_config.yaml', 'r') as f:
        train_config = yaml.safe_load(f)
    print(f"  ✓ 训练配置加载成功")
    print(f"     算法: {train_config['training']['algorithm']}")
    print(f"     总步数: {train_config['training']['total_timesteps']}")
    
except Exception as e:
    print(f"  ✗ 配置文件测试失败: {e}")

print("\n" + "="*60)
print("简化测试完成!")
print("="*60)

print("\n✅ 如果以上测试都通过，说明环境已正确安装!")
print("\n现在你可以运行训练:")
print("python scripts/train.py --total-timesteps 1000")
print("\n或运行完整训练（推荐）:")
print("python scripts/train.py --total-timesteps 50000 --use-gui")
print("\n注意：训练需要时间，请耐心等待。")
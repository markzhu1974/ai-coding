#!/usr/bin/env python3
"""
测试UR5机器人上下料任务环境安装
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("="*60)
print("UR5机器人上下料任务环境安装测试")
print("="*60)

# 测试1: 检查基本依赖
print("\n1. 检查Python版本...")
print(f"Python版本: {sys.version}")

# 测试2: 检查核心依赖
print("\n2. 检查核心依赖...")
dependencies = [
    ("numpy", "numpy"),
    ("gymnasium", "gymnasium"),
    ("pybullet", "pybullet"),
    ("torch", "torch"),
    ("stable-baselines3", "stable_baselines3"),
    ("PyYAML", "yaml"),
    ("matplotlib", "matplotlib"),
]

all_ok = True
for lib_name, import_name in dependencies:
    try:
        if import_name == "yaml":
            import yaml
        else:
            __import__(import_name)
        print(f"  ✓ {lib_name} 已安装")
    except ImportError as e:
        print(f"  ✗ {lib_name} 未安装: {e}")
        all_ok = False

if not all_ok:
    print("\n警告: 部分依赖未安装!")
    print("请运行以下命令安装缺失的依赖:")
    print("pip install torch stable-baselines3 pybullet gymnasium numpy matplotlib pandas PyYAML tqdm")
    print("\n或使用清华镜像源加速:")
    print("pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch stable-baselines3 pybullet")
else:
    print("\n所有核心依赖已安装!")

# 测试3: 测试环境创建
print("\n3. 测试环境创建...")
try:
    from envs.ur5_pickplace_env import UR5PickPlaceEnv
    
    # 创建环境（无GUI模式，更快）
    env = UR5PickPlaceEnv(render_mode='rgb_array', use_gui=False, max_steps=50)
    
    # 测试重置
    obs, info = env.reset()
    print(f"  ✓ 环境创建成功")
    print(f"    状态空间维度: {env.observation_space.shape}")
    print(f"    动作空间维度: {env.action_space.shape}")
    print(f"    初始状态形状: {obs.shape}")
    
    # 测试几步随机动作
    print(f"  ✓ 测试随机动作...")
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"    步骤 {i+1}: 奖励={reward:.3f}, 抓取={info.get('object_grabbed', False)}")
    
    env.close()
    print("  ✓ 环境测试完成")
    
except Exception as e:
    print(f"  ✗ 环境测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试4: 测试模型创建（如果所有依赖都安装了）
if all_ok:
    print("\n4. 测试模型创建...")
    try:
        import torch
        import torch.nn as nn
        from stable_baselines3 import PPO
        
        # 创建简单的环境用于测试模型创建
        from envs.ur5_pickplace_env import UR5PickPlaceEnv
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.monitor import Monitor
        
        def make_test_env():
            env = UR5PickPlaceEnv(render_mode='rgb_array', use_gui=False, max_steps=10)
            return Monitor(env)
        
        # 创建向量化环境
        env = make_vec_env(make_test_env, n_envs=1, seed=42)
        
        # 测试模型创建
        policy_kwargs = {
            'net_arch': [dict(pi=[64, 64], vf=[64, 64])],
            'activation_fn': nn.Tanh
        }
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=0.0003,
            n_steps=64,
            batch_size=32,
            n_epochs=1,
            gamma=0.99,
            policy_kwargs=policy_kwargs,
            verbose=0
        )
        
        print(f"  ✓ PPO模型创建成功")
        print(f"    参数数量: {sum(p.numel() for p in model.policy.parameters())}")
        
        env.close()
        
    except Exception as e:
        print(f"  ✗ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()

# 总结
print("\n" + "="*60)
print("安装测试完成")
print("="*60)

if all_ok:
    print("\n✅ 所有测试通过!")
    print("\n现在你可以运行训练:")
    print("python scripts/train.py --total-timesteps 1000 --use-gui")
    print("\n或运行快速演示:")
    print("python examples/quick_start.py")
else:
    print("\n⚠️  部分测试失败，请安装缺失的依赖")
    print("\n安装命令:")
    print("cd /home/zhulan/opencode/helloworld/robot_pickplace_demo")
    print("source venv_no_pip/bin/activate")
    print("pip install torch stable-baselines3")

print("\n" + "="*60)
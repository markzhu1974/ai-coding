#!/usr/bin/env python3
"""
UR5机器人上下料任务快速开始示例
这个脚本展示了如何使用本项目的基本功能
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print("="*60)
print("UR5机器人上下料任务快速开始示例")
print("="*60)

print("\n1. 项目结构:")
print("""
robot_pickplace_demo/
├── envs/                    # 自定义Gym环境
├── robots/                  # 机器人模型和控制器
├── scripts/                 # 训练和测试脚本
├── configs/                 # 配置文件
├── examples/                # 示例代码（当前目录）
└── README.md               # 项目说明
""")

print("\n2. 基本用法:")

print("""
# 安装依赖
pip install -r requirements.txt

# 简单训练（无GUI）
python scripts/train.py --total-timesteps 50000

# 带GUI的训练
python scripts/train.py --use-gui --total-timesteps 100000

# 测试模型
python scripts/test.py --model models/train_*/final_model.zip --render

# 可视化结果
python scripts/visualize.py --log-dir results/tensorboard/ --plots
""")

print("\n3. 代码示例:")

print("""
```python
from envs.ur5_pickplace_env import UR5PickPlaceEnv

# 创建环境
env = UR5PickPlaceEnv(render_mode='human', use_gui=True)

# 重置环境
obs, info = env.reset()
print(f"初始状态形状: {obs.shape}")
print(f"物体位置: {info['object_position']}")
print(f"目标位置: {info['target_position']}")

# 执行随机动作
for i in range(10):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"步骤 {i}: 奖励={reward:.2f}, 抓取={info['object_grabbed']}")

env.close()
```
""")

print("\n4. 配置文件位置:")
print("""
- 环境配置: configs/env_config.yaml
- 训练配置: configs/train_config.yaml
""")

print("\n5. 输出目录:")
print("""
- 模型保存: models/
- 训练日志: results/tensorboard/
- 评估结果: results/eval/
- 演示视频: videos/
""")

print("\n6. 常见任务:")

print("""
# 快速训练（1-2小时）
python scripts/train.py --total-timesteps 100000

# 批量评估
python scripts/test.py --model models/best_model.zip --episodes 10 --record

# 生成训练报告
python scripts/visualize.py --log-dir results/tensorboard/ --plots --demo
""")

print("\n7. 获取帮助:")
print("""
python scripts/train.py --help
python scripts/test.py --help
python scripts/visualize.py --help
""")

print("="*60)
print("更多信息请参考 README.md 文件")
print("="*60)
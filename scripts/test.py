#!/usr/bin/env python3
"""
UR5机器人上下料任务测试脚本
用于测试训练好的模型
"""

import os
import sys
import yaml
import argparse
import numpy as np
import cv2
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 尝试导入依赖
try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    
    from envs.ur5_pickplace_env import UR5PickPlaceEnv
    
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"导入依赖失败: {e}")
    print("请安装所需的依赖包:")
    print("pip install stable-baselines3 gymnasium pybullet numpy opencv-python")
    IMPORT_SUCCESS = False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="UR5机器人上下料任务测试")
    
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="模型文件路径"
    )
    
    parser.add_argument(
        "--env-config", 
        type=str, 
        default="configs/env_config.yaml",
        help="环境配置文件路径"
    )
    
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=5,
        help="测试的episode数量"
    )
    
    parser.add_argument(
        "--render", 
        action="store_true",
        help="渲染GUI界面"
    )
    
    parser.add_argument(
        "--record", 
        action="store_true",
        help="录制视频"
    )
    
    parser.add_argument(
        "--record-path", 
        type=str, 
        default="./videos/",
        help="视频保存路径"
    )
    
    parser.add_argument(
        "--fps", 
        type=int, 
        default=30,
        help="录制视频的帧率"
    )
    
    parser.add_argument(
        "--deterministic", 
        action="store_true",
        help="使用确定性策略"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def test_model(args):
    """测试模型"""
    print("="*60)
    print("UR5机器人上下料任务测试")
    print("="*60)
    print(f"模型路径: {args.model}")
    print(f"测试episode数量: {args.episodes}")
    print(f"渲染: {args.render}")
    print(f"录制视频: {args.record}")
    print(f"确定性策略: {args.deterministic}")
    
    # 加载环境配置
    env_config = load_config(args.env_config)
    
    # 更新环境配置
    env_config['env']['use_gui'] = args.render
    env_config['env']['render_mode'] = 'human' if args.render else 'rgb_array'
    
    # 创建环境
    env = UR5PickPlaceEnv(
        render_mode=env_config['env']['render_mode'],
        use_gui=env_config['env']['use_gui'],
        max_steps=env_config['env']['max_steps']
    )
    
    # 加载模型
    print(f"\n加载模型: {args.model}")
    model = PPO.load(args.model)
    
    # 创建视频录制器
    video_writer = None
    if args.record:
        os.makedirs(args.record_path, exist_ok=True)
        video_path = os.path.join(args.record_path, f"test_{os.path.basename(args.model).replace('.zip', '')}.mp4")
        
        # 获取图像尺寸
        test_frame = env.render()
        print(f"调试: test_frame 类型: {type(test_frame)}")
        if test_frame is not None:
            # PyBullet可能返回元组，需要提取RGB数组
            if isinstance(test_frame, tuple):
                print(f"警告: env.render() 返回了元组，长度为 {len(test_frame)}")
                # 假设第三个元素是RGB数组
                if len(test_frame) >= 3:
                    test_frame = test_frame[2]
                else:
                    test_frame = test_frame[0]
                print(f"调试: 提取后的类型: {type(test_frame)}")
            height, width = test_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, args.fps, (width, height))
            print(f"录制视频到: {video_path}")
    
    # 测试循环
    success_count = 0
    total_rewards = []
    episode_lengths = []
    
    for episode in range(args.episodes):
        print(f"\nEpisode {episode + 1}/{args.episodes}:")
        
        # 重置环境
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0
        
        print(f"  物体位置: {info['object_position']}")
        print(f"  目标位置: {info['target_position']}")
        
        frames = []
        
        while not done and not truncated:
            # 预测动作
            action, _ = model.predict(obs, deterministic=args.deterministic)
            
            # 执行动作
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            # 渲染
            if args.render:
                env.render()
            
            # 录制视频帧
            if args.record and video_writer is not None:
                frame = env.render()
                if frame is not None:
                    # 将RGB转换为BGR（OpenCV格式）
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr)
                    frames.append(frame_bgr)
            
            # 显示进度
            if step_count % 20 == 0:
                print(f"    步数: {step_count}, 累计奖励: {episode_reward:.2f}")
        
        # Episode结果
        if info.get('task_success', False):
            success_count += 1
            print(f"  ✓ 成功! 奖励: {episode_reward:.2f}, 步数: {step_count}")
        else:
            print(f"  ✗ 失败! 奖励: {episode_reward:.2f}, 步数: {step_count}")
        
        total_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        # 保存episode的视频
        if args.record and frames:
            episode_video_path = os.path.join(
                args.record_path, 
                f"episode_{episode+1}_{'success' if info.get('task_success', False) else 'fail'}.mp4"
            )
            if len(frames) > 0:
                height, width = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                episode_writer = cv2.VideoWriter(episode_video_path, fourcc, args.fps, (width, height))
                for frame in frames:
                    episode_writer.write(frame)
                episode_writer.release()
                print(f"  Episode视频保存到: {episode_video_path}")
    
    # 关闭视频录制器
    if video_writer is not None:
        video_writer.release()
    
    # 关闭环境
    env.close()
    
    # 打印测试结果
    print("\n" + "="*60)
    print("测试结果:")
    print("="*60)
    print(f"成功率: {success_count}/{args.episodes} ({100*success_count/args.episodes:.1f}%)")
    print(f"平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"平均步数: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"最大奖励: {np.max(total_rewards):.2f}")
    print(f"最小奖励: {np.min(total_rewards):.2f}")
    
    # 保存测试结果
    results = {
        'model_path': args.model,
        'episodes': args.episodes,
        'success_rate': float(success_count / args.episodes),
        'mean_reward': float(np.mean(total_rewards)),
        'std_reward': float(np.std(total_rewards)),
        'mean_steps': float(np.mean(episode_lengths)),
        'std_steps': float(np.std(episode_lengths)),
        'max_reward': float(np.max(total_rewards)),
        'min_reward': float(np.min(total_rewards)),
        'total_rewards': [float(r) for r in total_rewards],
        'episode_lengths': [int(l) for l in episode_lengths],
        'deterministic': args.deterministic
    }
    
    results_file = os.path.join(
        args.record_path if args.record else ".",
        f"test_results_{os.path.basename(args.model).replace('.zip', '')}.yaml"
    )
    
    with open(results_file, 'w') as f:
        yaml.dump(results, f)
    
    print(f"\n测试结果已保存到: {results_file}")
    
    return results


def main():
    """主函数"""
    if not IMPORT_SUCCESS:
        print("依赖导入失败，请先安装所需的包。")
        sys.exit(1)
    
    # 解析参数
    args = parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        sys.exit(1)
    
    # 运行测试
    test_model(args)


if __name__ == "__main__":
    main()
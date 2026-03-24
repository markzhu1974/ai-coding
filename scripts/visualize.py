#!/usr/bin/env python3
"""
UR5机器人上下料任务可视化脚本
用于分析训练结果和展示机器人性能
"""

import os
import sys
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 尝试导入依赖
try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.results_plotter import load_results, ts2xy
    
    from envs.ur5_pickplace_env import UR5PickPlaceEnv
    
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"导入依赖失败: {e}")
    print("请安装所需的依赖包:")
    print("pip install stable-baselines3 gymnasium matplotlib pandas numpy")
    IMPORT_SUCCESS = False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="UR5机器人上下料任务可视化")
    
    parser.add_argument(
        "--log-dir", 
        type=str, 
        required=True,
        help="训练日志目录路径"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        help="模型文件路径（用于展示演示）"
    )
    
    parser.add_argument(
        "--demo", 
        action="store_true",
        help="展示模型演示"
    )
    
    parser.add_argument(
        "--demo-episodes", 
        type=int, 
        default=3,
        help="演示的episode数量"
    )
    
    parser.add_argument(
        "--plots", 
        action="store_true",
        help="生成训练曲线图"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="./results/visualizations/",
        help="输出目录路径"
    )
    
    return parser.parse_args()


def plot_training_curves(log_dir: str, output_dir: str):
    """绘制训练曲线"""
    print(f"分析训练日志: {log_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 加载训练结果
        results = load_results(log_dir)
        
        if results is None or len(results) == 0:
            print("未找到训练日志数据")
            return
        
        # 转换数据
        timesteps, rewards = ts2xy(results, 'timesteps')
        _, lengths = ts2xy(results, 'ep_lengths')
        
        if len(timesteps) == 0:
            print("训练数据为空")
            return
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('UR5机器人上下料任务训练曲线', fontsize=16)
        
        # 1. 奖励曲线
        axes[0, 0].plot(timesteps, rewards, 'b-', alpha=0.6, linewidth=1)
        axes[0, 0].set_xlabel('时间步')
        axes[0, 0].set_ylabel('奖励')
        axes[0, 0].set_title('奖励曲线')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 计算移动平均
        window_size = min(100, len(rewards) // 10)
        if window_size > 1:
            rewards_ma = pd.Series(rewards).rolling(window=window_size).mean()
            axes[0, 0].plot(timesteps, rewards_ma, 'r-', linewidth=2, label=f'MA({window_size})')
            axes[0, 0].legend()
        
        # 2. Episode长度曲线
        axes[0, 1].plot(timesteps, lengths, 'g-', alpha=0.6, linewidth=1)
        axes[0, 1].set_xlabel('时间步')
        axes[0, 1].set_ylabel('Episode长度')
        axes[0, 1].set_title('Episode长度曲线')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 奖励直方图
        axes[0, 2].hist(rewards, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 2].set_xlabel('奖励')
        axes[0, 2].set_ylabel('频率')
        axes[0, 2].set_title('奖励分布')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Episode长度直方图
        axes[1, 0].hist(lengths, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].set_xlabel('Episode长度')
        axes[1, 0].set_ylabel('频率')
        axes[1, 0].set_title('Episode长度分布')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 累计奖励
        cumulative_rewards = np.cumsum(rewards)
        axes[1, 1].plot(timesteps, cumulative_rewards, 'purple', linewidth=2)
        axes[1, 1].set_xlabel('时间步')
        axes[1, 1].set_ylabel('累计奖励')
        axes[1, 1].set_title('累计奖励曲线')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 统计信息
        axes[1, 2].axis('off')
        stats_text = f"""
        训练统计:
        - 总时间步: {timesteps[-1] if len(timesteps) > 0 else 0}
        - 总episode数: {len(rewards)}
        - 平均奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}
        - 最大奖励: {np.max(rewards):.2f}
        - 最小奖励: {np.min(rewards):.2f}
        - 平均长度: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}
        """
        axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, 
                       verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(output_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线图已保存到: {plot_path}")
        
        # 显示图表
        plt.show()
        
        # 保存数据到CSV
        data = {
            'timesteps': timesteps,
            'rewards': rewards,
            'lengths': lengths
        }
        df = pd.DataFrame(data)
        csv_path = os.path.join(output_dir, 'training_data.csv')
        df.to_csv(csv_path, index=False)
        print(f"训练数据已保存到: {csv_path}")
        
    except Exception as e:
        print(f"绘制训练曲线时出错: {e}")


def show_demo(model_path: str, n_episodes: int = 3, output_dir: str = "./results/demos/"):
    """展示模型演示"""
    print(f"\n展示模型演示: {model_path}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    model = PPO.load(model_path)
    
    # 创建环境（显示GUI）
    env = UR5PickPlaceEnv(
        render_mode='human',
        use_gui=True,
        max_steps=200
    )
    
    results = []
    
    for episode in range(n_episodes):
        print(f"\n演示 Episode {episode + 1}/{n_episodes}:")
        
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0
        
        print(f"  物体位置: {info['object_position']}")
        print(f"  目标位置: {info['target_position']}")
        
        # 记录轨迹
        trajectory = {
            'ee_positions': [],
            'joint_positions': [],
            'rewards': [],
            'actions': [],
            'gripper_states': []
        }
        
        while not done and not truncated:
            # 预测动作
            action, _ = model.predict(obs, deterministic=True)
            
            # 执行动作
            obs, reward, done, truncated, info = env.step(action)
            
            # 渲染
            env.render()
            
            # 记录数据
            episode_reward += reward
            step_count += 1
            
            trajectory['ee_positions'].append(info.get('ee_position', [0, 0, 0]))
            trajectory['joint_positions'].append(obs[:6].tolist())  # 前6个是关节角度
            trajectory['rewards'].append(reward)
            trajectory['actions'].append(action.tolist())
            trajectory['gripper_states'].append(info.get('gripper_state', 0.0))
            
            # 显示进度
            if step_count % 20 == 0:
                print(f"    步数: {step_count}, 当前奖励: {reward:.2f}, 累计奖励: {episode_reward:.2f}")
        
        # Episode结果
        success = info.get('task_success', False)
        if success:
            print(f"  ✓ 成功完成! 总奖励: {episode_reward:.2f}, 步数: {step_count}")
        else:
            print(f"  ✗ 未完成! 总奖励: {episode_reward:.2f}, 步数: {step_count}")
        
        results.append({
            'episode': episode + 1,
            'success': success,
            'total_reward': float(episode_reward),
            'steps': step_count,
            'object_position': info.get('object_position', [0, 0, 0]),
            'target_position': info.get('target_position', [0, 0, 0])
        })
        
        # 保存轨迹数据
        traj_file = os.path.join(output_dir, f'trajectory_episode_{episode+1}.json')
        with open(traj_file, 'w') as f:
            json.dump(trajectory, f, indent=2)
        
        # 绘制轨迹图
        plot_trajectory(trajectory, episode + 1, output_dir)
    
    # 关闭环境
    env.close()
    
    # 保存演示结果
    results_file = os.path.join(output_dir, 'demo_results.yaml')
    with open(results_file, 'w') as f:
        yaml.dump(results, f)
    
    print(f"\n演示结果已保存到: {results_file}")
    
    # 打印摘要
    print("\n" + "="*50)
    print("演示摘要:")
    print("="*50)
    success_count = sum(1 for r in results if r['success'])
    print(f"成功率: {success_count}/{n_episodes} ({100*success_count/n_episodes:.1f}%)")
    
    total_rewards = [r['total_reward'] for r in results]
    print(f"平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    
    total_steps = [r['steps'] for r in results]
    print(f"平均步数: {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
    
    return results


def plot_trajectory(trajectory: dict, episode_num: int, output_dir: str):
    """绘制机器人轨迹图"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Episode {episode_num} - 机器人轨迹', fontsize=16)
        
        # 1. 末端执行器位置轨迹
        ee_positions = np.array(trajectory['ee_positions'])
        if len(ee_positions) > 0:
            axes[0, 0].plot(ee_positions[:, 0], ee_positions[:, 1], 'b-', linewidth=2, alpha=0.7)
            axes[0, 0].scatter(ee_positions[0, 0], ee_positions[0, 1], c='green', s=100, label='开始', marker='o')
            axes[0, 0].scatter(ee_positions[-1, 0], ee_positions[-1, 1], c='red', s=100, label='结束', marker='s')
            axes[0, 0].set_xlabel('X位置')
            axes[0, 0].set_ylabel('Y位置')
            axes[0, 0].set_title('末端执行器XY轨迹')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 奖励曲线
        rewards = trajectory['rewards']
        if len(rewards) > 0:
            axes[0, 1].plot(range(len(rewards)), rewards, 'g-', linewidth=2)
            axes[0, 1].fill_between(range(len(rewards)), rewards, alpha=0.3, color='green')
            axes[0, 1].set_xlabel('步数')
            axes[0, 1].set_ylabel('奖励')
            axes[0, 1].set_title('奖励曲线')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 关节角度变化
        joint_positions = np.array(trajectory['joint_positions'])
        if len(joint_positions) > 0 and joint_positions.shape[1] > 0:
            for i in range(min(6, joint_positions.shape[1])):
                axes[1, 0].plot(range(len(joint_positions)), joint_positions[:, i], 
                               label=f'关节{i+1}', alpha=0.7)
            axes[1, 0].set_xlabel('步数')
            axes[1, 0].set_ylabel('关节角度 (rad)')
            axes[1, 0].set_title('关节角度变化')
            axes[1, 0].legend(loc='upper right', fontsize='small')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 夹爪状态
        gripper_states = trajectory['gripper_states']
        if len(gripper_states) > 0:
            axes[1, 1].plot(range(len(gripper_states)), gripper_states, 'orange', linewidth=2)
            axes[1, 1].set_xlabel('步数')
            axes[1, 1].set_ylabel('夹爪状态')
            axes[1, 1].set_title('夹爪开合状态 (0=闭合, 1=打开)')
            axes[1, 1].set_ylim(-0.1, 1.1)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        traj_plot_path = os.path.join(output_dir, f'trajectory_episode_{episode_num}.png')
        plt.savefig(traj_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"绘制轨迹图时出错: {e}")


def analyze_training_results(log_dir: str, output_dir: str):
    """分析训练结果"""
    print(f"\n分析训练结果: {log_dir}")
    
    # 查找评估结果
    eval_dir = os.path.join(log_dir, 'eval')
    if os.path.exists(eval_dir):
        print(f"找到评估目录: {eval_dir}")
        
        # 查找评估日志文件
        eval_logs = []
        for root, dirs, files in os.walk(eval_dir):
            for file in files:
                if file.endswith('.csv'):
                    eval_logs.append(os.path.join(root, file))
        
        if eval_logs:
            print(f"找到 {len(eval_logs)} 个评估日志文件")
            
            # 读取并分析评估数据
            all_eval_data = []
            for eval_log in eval_logs:
                try:
                    df = pd.read_csv(eval_log)
                    if 'r' in df.columns:  # 奖励列
                        all_eval_data.append(df)
                except Exception as e:
                    print(f"读取评估日志失败 {eval_log}: {e}")
            
            if all_eval_data:
                # 合并评估数据
                combined_df = pd.concat(all_eval_data, ignore_index=True)
                
                # 绘制评估曲线
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # 评估奖励曲线
                if 'r' in combined_df.columns and 'l' in combined_df.columns:
                    axes[0].plot(combined_df['l'], combined_df['r'], 'b-', alpha=0.6, linewidth=1)
                    axes[0].set_xlabel('时间步')
                    axes[0].set_ylabel('评估奖励')
                    axes[0].set_title('评估奖励曲线')
                    axes[0].grid(True, alpha=0.3)
                
                # 评估成功率
                if 'success' in combined_df.columns:
                    success_rate = combined_df['success'].mean()
                    axes[1].bar(['成功率'], [success_rate], color=['green' if success_rate > 0.5 else 'red'])
                    axes[1].set_ylim(0, 1)
                    axes[1].set_ylabel('成功率')
                    axes[1].set_title(f'评估成功率: {success_rate:.1%}')
                    axes[1].grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                
                # 保存评估图表
                eval_plot_path = os.path.join(output_dir, 'evaluation_analysis.png')
                plt.savefig(eval_plot_path, dpi=300, bbox_inches='tight')
                print(f"评估分析图已保存到: {eval_plot_path}")
                
                plt.show()
        
        # 查找最佳模型
        best_model_dir = os.path.join(eval_dir, 'best_model')
        if os.path.exists(best_model_dir):
            print(f"找到最佳模型目录: {best_model_dir}")
    
    # 查找训练摘要
    summary_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.endswith('summary.yaml') or file.endswith('summary.yml'):
                summary_files.append(os.path.join(root, file))
    
    if summary_files:
        print(f"\n找到 {len(summary_files)} 个训练摘要文件")
        for summary_file in summary_files:
            try:
                with open(summary_file, 'r') as f:
                    summary = yaml.safe_load(f)
                print(f"\n训练摘要 ({os.path.basename(summary_file)}):")
                print(f"  算法: {summary.get('training', {}).get('algorithm', 'N/A')}")
                print(f"  总步数: {summary.get('training', {}).get('total_timesteps', 'N/A')}")
                
                eval_results = summary.get('evaluation', {})
                if eval_results:
                    print(f"  评估成功率: {eval_results.get('success_rate', 0):.1%}")
                    print(f"  平均奖励: {eval_results.get('mean_reward', 0):.2f}")
            except Exception as e:
                print(f"读取训练摘要失败 {summary_file}: {e}")


def main():
    """主函数"""
    if not IMPORT_SUCCESS:
        print("依赖导入失败，请先安装所需的包。")
        sys.exit(1)
    
    # 解析参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 生成训练曲线图
    if args.plots:
        plot_training_curves(args.log_dir, args.output)
    
    # 分析训练结果
    analyze_training_results(args.log_dir, args.output)
    
    # 展示模型演示
    if args.demo and args.model:
        if os.path.exists(args.model):
            show_demo(args.model, args.demo_episodes, 
                     os.path.join(args.output, 'demos'))
        else:
            print(f"错误: 模型文件不存在: {args.model}")
    
    print("\n可视化完成!")


if __name__ == "__main__":
    main()
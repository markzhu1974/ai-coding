#!/usr/bin/env python3
"""
UR5机器人上下料任务的强化学习训练脚本
使用Stable-Baselines3的PPO算法
"""

import os
import sys
import yaml
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 尝试导入依赖
try:
    import gymnasium as gym
    import torch
    import torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import (
        EvalCallback,
        CheckpointCallback,
        CallbackList,
        ProgressBarCallback
    )
    from stable_baselines3.common.monitor import Monitor
    
    from envs.ur5_pickplace_env import UR5PickPlaceEnv
    
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"导入依赖失败: {e}")
    print("请安装所需的依赖包:")
    print("pip install stable-baselines3 gymnasium pybullet numpy PyYAML torch")
    IMPORT_SUCCESS = False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="UR5机器人上下料任务训练")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/train_config.yaml",
        help="训练配置文件路径"
    )
    
    parser.add_argument(
        "--env-config", 
        type=str, 
        default="configs/env_config.yaml",
        help="环境配置文件路径"
    )
    
    parser.add_argument(
        "--total-timesteps", 
        type=int, 
        default=None,
        help="总训练步数（覆盖配置文件）"
    )
    
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=None,
        help="学习率（覆盖配置文件）"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="随机种子（覆盖配置文件）"
    )
    
    parser.add_argument(
        "--use-gui", 
        action="store_true",
        help="训练时显示GUI（会减慢训练速度）"
    )
    
    parser.add_argument(
        "--eval-only", 
        action="store_true",
        help="仅评估模型，不训练"
    )
    
    parser.add_argument(
        "--model-path", 
        type=str, 
        default=None,
        help="要加载的模型路径（用于评估或继续训练）"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_env(config: dict, env_config: dict, use_gui: bool = False):
    """创建训练环境"""
    # 更新环境配置
    env_config['env']['use_gui'] = use_gui
    
    def make_env():
        """环境创建函数"""
        env = UR5PickPlaceEnv(
            render_mode=env_config['env']['render_mode'],
            use_gui=env_config['env']['use_gui'],
            max_steps=env_config['env']['max_steps'],
            workspace_bounds=env_config['task']['workspace_bounds']
        )
        env = Monitor(env)  # 添加监控器
        return env
    
    # 创建向量化环境
    n_envs = config['training']['n_envs']
    env = make_vec_env(
        make_env,
        n_envs=n_envs,
        seed=config['training'].get('seed', 42)
    )
    
    # 可选：对环境进行标准化
    if config['training'].get('normalize', False):
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    return env


def create_model(env, config: dict, model_path: str = None):
    """创建或加载模型"""
    # 激活函数映射
    activation_mapping = {
        'tanh': nn.Tanh,
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'elu': nn.ELU,
        'selu': nn.SELU,
        'sigmoid': nn.Sigmoid
    }
    
    activation_name = config['policy']['activation_fn']
    if activation_name in activation_mapping:
        activation_fn = activation_mapping[activation_name]
    else:
        print(f"警告: 未知的激活函数 '{activation_name}'，使用默认的 tanh")
        activation_fn = nn.Tanh
    
    policy_kwargs = {
        'net_arch': dict(pi=config['policy']['network_arch'], 
                         vf=config['policy']['network_arch']),
        'activation_fn': activation_fn
    }
    
    if model_path and os.path.exists(model_path):
        # 加载现有模型
        print(f"从 {model_path} 加载模型...")
        model = PPO.load(
            model_path,
            env=env,
            custom_objects=policy_kwargs
        )
    else:
        # 创建新模型
        print("创建新模型...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config['ppo']['learning_rate'],
            n_steps=config['ppo']['n_steps'],
            batch_size=config['ppo']['batch_size'],
            n_epochs=config['ppo']['n_epochs'],
            gamma=config['ppo']['gamma'],
            gae_lambda=config['ppo']['gae_lambda'],
            clip_range=config['ppo']['clip_range'],
            clip_range_vf=config['ppo']['clip_range_vf'],
            ent_coef=config['ppo']['ent_coef'],
            vf_coef=config['ppo']['vf_coef'],
            max_grad_norm=config['ppo']['max_grad_norm'],
            tensorboard_log=config['logging']['tensorboard_log'],
            policy_kwargs=policy_kwargs,
            verbose=config['logging']['verbose'],
            seed=config['training'].get('seed', 42)
        )
    
    return model


def create_callbacks(env, config: dict, save_path: str):
    """创建回调函数"""
    callbacks = []
    
    # 进度条回调
    if config['callbacks'].get('use_progress_bar', True):
        try:
            # 检查是否安装了所需的包
            import tqdm
            import rich
            callbacks.append(ProgressBarCallback())
            print("进度条回调已启用")
        except ImportError:
            print("警告: 缺少进度条所需的包 (tqdm 或 rich)，跳过进度条回调")
            print("要启用进度条，请运行: pip install rich")
    
    # 检查点回调
    if config['callbacks'].get('use_checkpoint_callback', True):
        checkpoint_callback = CheckpointCallback(
            save_freq=config['logging']['save_freq'],
            save_path=save_path,
            name_prefix='ur5_pickplace'
        )
        callbacks.append(checkpoint_callback)
    
    # 评估回调
    if config['callbacks'].get('use_eval_callback', True):
        eval_callback = EvalCallback(
            env,
            best_model_save_path=os.path.join(save_path, 'best_model'),
            log_path=config['evaluation']['eval_log_path'],
            eval_freq=config['evaluation']['eval_freq'],
            n_eval_episodes=config['evaluation']['n_eval_episodes'],
            deterministic=True,
            render=False,
            verbose=1
        )
        callbacks.append(eval_callback)
        print("评估回调已启用")
    
    return CallbackList(callbacks) if callbacks else None


def train_model(model, config: dict, callbacks=None):
    """训练模型"""
    total_timesteps = config['training']['total_timesteps']
    
    print(f"开始训练，总步数: {total_timesteps}")
    print(f"算法: {config['training']['algorithm']}")
    print(f"策略网络: {config['policy']['network_arch']}")
    print(f"学习率: {config['ppo']['learning_rate']}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=False
    )
    
    return model


def evaluate_model(model, env_config: dict, n_episodes: int = 5, use_gui: bool = False):
    """评估模型性能"""
    print(f"\n评估模型，运行 {n_episodes} 个episode...")
    
    # 创建评估环境（避免GUI冲突，使用rgb_array模式）
    # 注意：PyBullet只允许一个GUI连接，所以评估时通常不使用GUI
    env = UR5PickPlaceEnv(
        render_mode='rgb_array',
        use_gui=False,
        max_steps=env_config['env']['max_steps']
    )
    
    success_count = 0
    total_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0
        
        print(f"\nEpisode {episode + 1}/{n_episodes}:")
        
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            if use_gui:
                env.render()
        
        if info.get('task_success', False):
            success_count += 1
            print(f"  成功! 奖励: {episode_reward:.2f}, 步数: {step_count}")
        else:
            print(f"  失败! 奖励: {episode_reward:.2f}, 步数: {step_count}")
        
        total_rewards.append(episode_reward)
        episode_lengths.append(step_count)
    
    env.close()
    
    # 打印评估结果
    print("\n" + "="*50)
    print("评估结果:")
    print(f"成功率: {success_count}/{n_episodes} ({100*success_count/n_episodes:.1f}%)")
    print(f"平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"平均步数: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print("="*50)
    
    return {
        'success_rate': success_count / n_episodes,
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_steps': np.mean(episode_lengths),
        'std_steps': np.std(episode_lengths)
    }


def main():
    """主函数"""
    if not IMPORT_SUCCESS:
        print("依赖导入失败，请先安装所需的包。")
        sys.exit(1)
    
    # 解析参数
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    env_config = load_config(args.env_config)
    
    # 用命令行参数覆盖配置
    if args.total_timesteps is not None:
        config['training']['total_timesteps'] = args.total_timesteps
    if args.learning_rate is not None:
        config['ppo']['learning_rate'] = args.learning_rate
    if args.seed is not None:
        config['training']['seed'] = args.seed
    
    # 设置随机种子
    seed = config['training'].get('seed', 42)
    np.random.seed(seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(config['logging']['save_path'], f"train_{timestamp}")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(config['logging']['tensorboard_log'], exist_ok=True)
    os.makedirs(config['evaluation']['eval_log_path'], exist_ok=True)
    
    print("="*60)
    print("UR5机器人上下料任务强化学习训练")
    print("="*60)
    print(f"项目根目录: {project_root}")
    print(f"输出目录: {save_path}")
    print(f"随机种子: {seed}")
    
    # 创建环境
    print("\n创建环境...")
    env = create_env(config, env_config, use_gui=args.use_gui)
    
    # 创建或加载模型
    model = create_model(env, config, model_path=args.model_path)
    
    if args.eval_only:
        # 仅评估模式
        print("\n进入评估模式...")
        eval_results = evaluate_model(
            model, 
            env_config, 
            n_episodes=5, 
            use_gui=args.use_gui
        )
        
        # 保存评估结果
        eval_file = os.path.join(save_path, 'evaluation_results.yaml')
        with open(eval_file, 'w') as f:
            yaml.dump(eval_results, f)
        
        print(f"\n评估结果已保存到: {eval_file}")
    else:
        # 训练模式
        # 创建回调函数
        callbacks = create_callbacks(env, config, save_path)
        
        # 训练模型
        print("\n开始训练...")
        model = train_model(model, config, callbacks)
        
        # 保存最终模型
        final_model_path = os.path.join(save_path, 'final_model.zip')
        model.save(final_model_path)
        print(f"\n最终模型已保存到: {final_model_path}")
        
        # 评估最终模型
        print("\n评估最终模型...")
        eval_results = evaluate_model(
            model, 
            env_config, 
            n_episodes=5, 
            use_gui=True  # 评估时显示GUI
        )
        
        # 保存训练结果摘要
        summary = {
            'training': {
                'algorithm': config['training']['algorithm'],
                'total_timesteps': config['training']['total_timesteps'],
                'seed': seed,
                'save_path': save_path
            },
            'evaluation': eval_results,
            'hyperparameters': config['ppo']
        }
        
        summary_file = os.path.join(save_path, 'training_summary.yaml')
        with open(summary_file, 'w') as f:
            yaml.dump(summary, f)
        
        print(f"\n训练摘要已保存到: {summary_file}")
    
    # 清理
    env.close()
    print("\n训练完成!")


if __name__ == "__main__":
    main()
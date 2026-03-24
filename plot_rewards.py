#!/usr/bin/env python3
"""
Plot training rewards from tensorboard logs.
"""
import os
import sys
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def main():
    log_dir = "results/tensorboard/PPO_0"
    if not os.path.exists(log_dir):
        print(f"Log directory not found: {log_dir}")
        sys.exit(1)
    
    ea = EventAccumulator(log_dir)
    ea.Reload()
    
    # Get rollout rewards
    if 'rollout/ep_rew_mean' not in ea.Tags()['scalars']:
        print("No rollout rewards found")
        return
    
    rewards = ea.Scalars('rollout/ep_rew_mean')
    steps = [r.step for r in rewards]
    values = [r.value for r in rewards]
    
    # Get eval rewards if available
    eval_steps = []
    eval_values = []
    if 'eval/mean_reward' in ea.Tags()['scalars']:
        eval_rewards = ea.Scalars('eval/mean_reward')
        eval_steps = [r.step for r in eval_rewards]
        eval_values = [r.value for r in eval_rewards]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, values, label='Rollout mean reward', alpha=0.7)
    if eval_steps:
        plt.scatter(eval_steps, eval_values, color='red', label='Evaluation mean reward', s=30, zorder=5)
    
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Episode Reward')
    plt.title('UR5 Pick-and-Place Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save
    os.makedirs('results/plots', exist_ok=True)
    output_path = 'results/plots/training_rewards.png'
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    
    # Show summary
    if len(values) > 0:
        print(f"Latest rollout reward: {values[-1]:.2f} at step {steps[-1]}")
        print(f"Best rollout reward: {max(values):.2f} at step {steps[values.index(max(values))]}")
    if eval_values:
        print(f"Best eval reward: {max(eval_values):.2f} at step {eval_steps[eval_values.index(max(eval_values))]}")

if __name__ == '__main__':
    main()
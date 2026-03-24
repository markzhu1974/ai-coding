#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
import yaml
with open('configs/env_config_easy.yaml', 'r') as f:
    config = yaml.safe_load(f)
print('Config loaded:', config.keys())
print('Workspace bounds:', config['task']['workspace_bounds'])

from envs.ur5_pickplace_env import UR5PickPlaceEnv
env = UR5PickPlaceEnv(
    render_mode='rgb_array',
    use_gui=False,
    max_steps=50,
    workspace_bounds=config['task']['workspace_bounds']
)
print('Env created')
obs, info = env.reset()
print('Reset ok, obs shape:', obs.shape)
print('Object position:', info['object_position'])
print('Target position:', info['target_position'])
env.close()
print('Test passed.')
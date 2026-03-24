#!/usr/bin/env python3
"""
UR5机器人上下料强化学习仿真环境安装配置
"""

from setuptools import setup, find_packages
import os

# 读取README文件
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# 读取requirements.txt
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='robot_pickplace_demo',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='UR5机器人上下料任务的强化学习仿真环境',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/robot_pickplace_demo',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Robotics',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=22.0.0',
            'isort>=5.10.0',
            'flake8>=4.0.0',
            'mypy>=0.991',
        ],
        'docs': [
            'sphinx>=5.0.0',
            'sphinx-rtd-theme>=1.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'ur5-train=scripts.train:main',
            'ur5-test=scripts.test:main',
            'ur5-visualize=scripts.visualize:main',
        ],
    },
    package_data={
        'robots': ['urdf/*.urdf'],
        'configs': ['*.yaml'],
    },
    include_package_data=True,
    keywords=[
        'robotics',
        'reinforcement-learning',
        'simulation',
        'ur5',
        'pick-and-place',
        'pybullet',
        'stable-baselines3',
    ],
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/robot_pickplace_demo/issues',
        'Source': 'https://github.com/yourusername/robot_pickplace_demo',
        'Documentation': 'https://github.com/yourusername/robot_pickplace_demo/wiki',
    },
)
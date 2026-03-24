import numpy as np
import pybullet as p
import pybullet_data
import os
from typing import Tuple, List, Optional


class UR5Robot:
    """
    UR5机器人封装类，提供机器人控制和状态查询接口
    """
    
    def __init__(self, physics_client_id: int, use_gui: bool = False):
        """
        初始化UR5机器人
        
        Args:
            physics_client_id: PyBullet物理客户端ID
            use_gui: 是否使用GUI模式
        """
        self.physics_client_id = physics_client_id
        self.use_gui = use_gui
        
        # 机器人ID和关节信息
        self.robot_id = None
        self.num_joints = 0
        self.joint_indices = []
        self.joint_names = []
        self.joint_lower_limits = []
        self.joint_upper_limits = []
        self.joint_ranges = []
        
        # 末端执行器和夹爪信息
        self.ee_link_index = None
        self.gripper_joint_index = None
        
        # 初始化机器人
        self._load_robot()
        self._setup_joint_info()
        
    def _load_robot(self):
        """加载UR5机器人模型"""
        # 设置PyBullet数据路径
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 加载地面
        p.loadURDF("plane.urdf", [0, 0, 0], physicsClientId=self.physics_client_id)
        
        # 加载UR5机器人
        urdf_path = os.path.join(os.path.dirname(__file__), "urdf", "ur5.urdf")
        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            physicsClientId=self.physics_client_id
        )
        
        # 创建桌子
        table_urdf = os.path.join(pybullet_data.getDataPath(), "table", "table.urdf")
        if os.path.exists(table_urdf):
            self.table_id = p.loadURDF(
                table_urdf,
                basePosition=[0.5, 0, 0],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.physics_client_id
            )
        else:
            # 创建简单的桌子作为替代
            table_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[0.4, 0.4, 0.02],
                physicsClientId=self.physics_client_id
            )
            self.table_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=table_shape,
                basePosition=[0.5, 0, 0.02],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.physics_client_id
            )
    
    def _setup_joint_info(self):
        """设置关节信息"""
        self.num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client_id)
        
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client_id)
            
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            
            # 只处理旋转关节和棱柱关节
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.joint_indices.append(i)
                self.joint_names.append(joint_name)
                self.joint_lower_limits.append(joint_info[8])
                self.joint_upper_limits.append(joint_info[9])
                self.joint_ranges.append(joint_info[9] - joint_info[8])
                
                # 检查是否为末端执行器或夹爪关节
                if 'ee' in joint_name.lower() or 'wrist_3' in joint_name.lower():
                    self.ee_link_index = i
                elif 'gripper' in joint_name.lower():
                    self.gripper_joint_index = i
        
        # 如果没有明确找到末端执行器，使用最后一个关节
        if self.ee_link_index is None and self.joint_indices:
            self.ee_link_index = self.joint_indices[-1]
        
        print(f"加载机器人成功，共有 {len(self.joint_indices)} 个可控关节")
        print(f"关节名称: {self.joint_names}")
    
    def get_joint_positions(self) -> np.ndarray:
        """
        获取当前关节位置
        
        Returns:
            关节位置数组
        """
        joint_positions = []
        for joint_idx in self.joint_indices:
            joint_state = p.getJointState(
                self.robot_id, 
                joint_idx,
                physicsClientId=self.physics_client_id
            )
            joint_positions.append(joint_state[0])
        
        return np.array(joint_positions, dtype=np.float32)
    
    def get_joint_velocities(self) -> np.ndarray:
        """
        获取当前关节速度
        
        Returns:
            关节速度数组
        """
        joint_velocities = []
        for joint_idx in self.joint_indices:
            joint_state = p.getJointState(
                self.robot_id, 
                joint_idx,
                physicsClientId=self.physics_client_id
            )
            joint_velocities.append(joint_state[1])
        
        return np.array(joint_velocities, dtype=np.float32)
    
    def get_ee_position(self) -> np.ndarray:
        """
        获取末端执行器位置
        
        Returns:
            末端执行器位置 [x, y, z]
        """
        if self.ee_link_index is None:
            raise ValueError("未找到末端执行器链接")
        
        link_state = p.getLinkState(
            self.robot_id,
            self.ee_link_index,
            physicsClientId=self.physics_client_id
        )
        return np.array(link_state[0], dtype=np.float32)
    
    def get_ee_orientation(self) -> np.ndarray:
        """
        获取末端执行器姿态（四元数）
        
        Returns:
            四元数姿态 [x, y, z, w]
        """
        if self.ee_link_index is None:
            raise ValueError("未找到末端执行器链接")
        
        link_state = p.getLinkState(
            self.robot_id,
            self.ee_link_index,
            physicsClientId=self.physics_client_id
        )
        return np.array(link_state[1], dtype=np.float32)
    
    def get_gripper_state(self) -> float:
        """
        获取夹爪状态
        
        Returns:
            夹爪开合程度（0-1之间，0为完全闭合，1为完全打开）
        """
        if self.gripper_joint_index is None:
            return 0.0
        
        joint_state = p.getJointState(
            self.robot_id,
            self.gripper_joint_index,
            physicsClientId=self.physics_client_id
        )
        
        # 归一化到0-1范围
        gripper_pos = joint_state[0]
        lower_limit = self.joint_lower_limits[self.joint_indices.index(self.gripper_joint_index)]
        upper_limit = self.joint_upper_limits[self.joint_indices.index(self.gripper_joint_index)]
        
        normalized = (gripper_pos - lower_limit) / (upper_limit - lower_limit)
        return np.clip(normalized, 0.0, 1.0)
    
    def set_joint_positions(self, positions: np.ndarray, max_force: float = 50.0):
        """
        设置关节目标位置（位置控制）
        
        Args:
            positions: 目标关节位置数组
            max_force: 最大控制力
        """
        if len(positions) != len(self.joint_indices):
            raise ValueError(f"位置数组长度 {len(positions)} 与关节数量 {len(self.joint_indices)} 不匹配")
        
        for i, joint_idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=positions[i],
                force=max_force,
                physicsClientId=self.physics_client_id
            )
    
    def set_joint_velocities(self, velocities: np.ndarray, max_force: float = 50.0):
        """
        设置关节目标速度（速度控制）
        
        Args:
            velocities: 目标关节速度数组
            max_force: 最大控制力
        """
        if len(velocities) != len(self.joint_indices):
            raise ValueError(f"速度数组长度 {len(velocities)} 与关节数量 {len(self.joint_indices)} 不匹配")
        
        for i, joint_idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.VELOCITY_CONTROL,
                targetVelocity=velocities[i],
                force=max_force,
                physicsClientId=self.physics_client_id
            )
    
    def set_gripper(self, gripper_value: float, max_force: float = 10.0):
        """
        设置夹爪开合
        
        Args:
            gripper_value: 夹爪开合程度（0-1之间）
            max_force: 最大控制力
        """
        if self.gripper_joint_index is None:
            return
        
        # 将0-1值映射到关节限制范围
        gripper_value = np.clip(gripper_value, 0.0, 1.0)
        joint_idx = self.gripper_joint_index
        idx_in_list = self.joint_indices.index(joint_idx)
        
        lower_limit = self.joint_lower_limits[idx_in_list]
        upper_limit = self.joint_upper_limits[idx_in_list]
        
        target_position = lower_limit + gripper_value * (upper_limit - lower_limit)
        
        p.setJointMotorControl2(
            self.robot_id,
            joint_idx,
            p.POSITION_CONTROL,
            targetPosition=target_position,
            force=max_force,
            physicsClientId=self.physics_client_id
        )
    
    def reset(self, joint_positions: Optional[np.ndarray] = None):
        """
        重置机器人到初始状态
        
        Args:
            joint_positions: 可选的目标关节位置，如果为None则使用中间位置
        """
        if joint_positions is None:
            # 使用关节范围的中间位置
            temp_positions = []
            for lower, upper in zip(self.joint_lower_limits, self.joint_upper_limits):
                temp_positions.append((lower + upper) / 2.0)
            joint_positions = np.array(temp_positions, dtype=np.float32)
        
        # 确保joint_positions不是None
        assert joint_positions is not None, "关节位置不能为None"
        
        # 重置关节位置
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(
                self.robot_id,
                joint_idx,
                targetValue=joint_positions[i],
                physicsClientId=self.physics_client_id
            )
        
        # 重置夹爪为打开状态
        self.set_gripper(1.0)
        
        # 禁用关节电机以允许重力作用
        for joint_idx in self.joint_indices:
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0,
                physicsClientId=self.physics_client_id
            )
        
        # 执行几步模拟让机器人稳定
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.physics_client_id)
    
    def check_collision(self, other_body_id: Optional[int] = None) -> bool:
        """
        检查机器人是否发生碰撞
        
        Args:
            other_body_id: 可选的检查与特定物体的碰撞
            
        Returns:
            是否发生碰撞
        """
        if other_body_id is None:
            # 检查与所有物体的碰撞
            contact_points = p.getContactPoints(
                bodyA=self.robot_id,
                physicsClientId=self.physics_client_id
            )
        else:
            # 检查与特定物体的碰撞
            contact_points = p.getContactPoints(
                bodyA=self.robot_id,
                bodyB=other_body_id,
                physicsClientId=self.physics_client_id
            )
        
        return len(contact_points) > 0
    
    def inverse_kinematics(self, target_position: np.ndarray, target_orientation: Optional[np.ndarray] = None) -> np.ndarray:
        """
        逆运动学计算
        
        Args:
            target_position: 目标位置 [x, y, z]
            target_orientation: 可选的目标姿态（四元数）
            
        Returns:
            关节位置数组
        """
        if target_orientation is None:
            # 使用当前末端执行器姿态
            target_orientation = self.get_ee_orientation()
        
        # 使用PyBullet的逆运动学求解器
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_index,
            target_position,
            targetOrientation=target_orientation,
            lowerLimits=self.joint_lower_limits,
            upperLimits=self.joint_upper_limits,
            jointRanges=self.joint_ranges,
            physicsClientId=self.physics_client_id
        )
        
        # 只返回可控关节的位置
        return np.array(joint_positions[:len(self.joint_indices)], dtype=np.float32)
    
    def forward_kinematics(self, joint_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        正向运动学计算
        
        Args:
            joint_positions: 关节位置数组
            
        Returns:
            (位置, 姿态) 元组
        """
        # 临时设置关节位置并计算正向运动学
        temp_positions = []
        for i, joint_idx in enumerate(self.joint_indices):
            if i < len(joint_positions):
                temp_positions.append(joint_positions[i])
            else:
                temp_positions.append(0.0)
        
        # 计算正向运动学
        joint_states = p.calculateForwardKinematics(
            self.robot_id,
            self.joint_indices,
            temp_positions,
            physicsClientId=self.physics_client_id
        )
        
        # 获取末端执行器位置和姿态
        if self.ee_link_index is not None and self.ee_link_index in self.joint_indices:
            ee_index = self.joint_indices.index(self.ee_link_index)
            position = np.array(joint_states[ee_index * 3: ee_index * 3 + 3], dtype=np.float32)
            orientation = np.array(joint_states[ee_index * 3 + 3: ee_index * 3 + 7], dtype=np.float32)
            return position, orientation
        
        return np.zeros(3, dtype=np.float32), np.array([0, 0, 0, 1], dtype=np.float32)
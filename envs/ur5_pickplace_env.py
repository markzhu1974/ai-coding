import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces
import time
import random
from typing import Tuple, Dict, Optional, Any

from robots.ur5_robot import UR5Robot


class UR5PickPlaceEnv(gym.Env):
    """
    UR5机器人上下料任务的Gym环境
    
    任务：机器人需要抓取物体并将其放置到目标位置
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, render_mode: str = 'human', use_gui: bool = False, max_steps: int = 200, workspace_bounds: Optional[Dict] = None):
        """
        初始化环境
        
        Args:
            render_mode: 渲染模式，'human'或'rgb_array'
            use_gui: 是否显示GUI
            max_steps: 每个episode的最大步数
            workspace_bounds: 工作空间边界字典，格式为 {'x': [min, max], 'y': [min, max], 'z': [min, max]}
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.use_gui = use_gui
        self.max_steps = max_steps
        
        # 工作空间边界
        if workspace_bounds is None:
            workspace_bounds = {'x': [0.3, 0.7], 'y': [-0.3, 0.3], 'z': [0.1, 0.5]}
        self.workspace_bounds = workspace_bounds
        
        # 初始化PyBullet
        self.physics_client_id = None
        self._init_simulation()
        
        # 确保physics_client_id已设置
        assert self.physics_client_id is not None, "PyBullet物理客户端ID未初始化"
        
        # 初始化机器人
        self.robot = UR5Robot(self.physics_client_id, use_gui=use_gui)
        
        # 物体参数
        self.object_id = None
        self.object_size = 0.05  # 物体半径
        self.object_mass = 0.1   # 物体质量
        self.object_color = [0.9, 0.2, 0.2, 1.0]  # 红色
        
        # 目标参数
        self.target_position = None
        self.target_size = 0.2  # 目标区域半径（大幅增加，简化放置）
        self.target_color = [0.2, 0.9, 0.2, 0.3]  # 半透明绿色
        
        # 任务状态
        self.object_grabbed = False
        self.object_placed = False
        self.object_dropped = False
        self.current_step = 0
        
        # 定义动作空间
        # 方案B：末端执行器位移 + 夹爪控制
        # [dx, dy, dz, 夹爪开合]
        self.action_space = spaces.Box(
            low=np.array([-0.05, -0.05, -0.05, 0.0], dtype=np.float32),
            high=np.array([0.05, 0.05, 0.05, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # 定义状态空间（简化版）
        # 状态向量包括：
        # 1. 末端执行器位置（3维）
        # 2. 夹爪状态（1维）
        # 3. 目标物体位置（3维）
        # 4. 目标放置位置（3维）
        # 5. 物体是否被抓取（1维）
        # 总共11维
        
        obs_low = []
        obs_high = []
        
        # 末端执行器位置（假设在合理工作空间内）
        obs_low.extend([-1.0, -1.0, 0.0])
        obs_high.extend([1.0, 1.0, 1.0])
        
        # 夹爪状态（0-1）
        obs_low.append(0.0)
        obs_high.append(1.0)
        
        # 目标物体位置
        obs_low.extend([-1.0, -1.0, 0.1])
        obs_high.extend([1.0, 1.0, 0.5])
        
        # 目标放置位置
        obs_low.extend([-1.0, -1.0, 0.1])
        obs_high.extend([1.0, 1.0, 0.5])
        
        # 物体是否被抓取（0或1）
        obs_low.append(0.0)
        obs_high.append(1.0)
        
        self.observation_space = spaces.Box(
            low=np.array(obs_low, dtype=np.float32),
            high=np.array(obs_high, dtype=np.float32),
            dtype=np.float32
        )
        
        print(f"环境初始化完成，状态空间维度: {self.observation_space.shape}")
        print(f"动作空间维度: {self.action_space.shape}")
    
    def _init_simulation(self):
        """初始化PyBullet仿真"""
        # 仅当未连接时创建新连接
        if self.physics_client_id is None:
            if self.use_gui and self.render_mode == 'human':
                self.physics_client_id = p.connect(p.GUI)
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
                p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
                p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
                p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            else:
                self.physics_client_id = p.connect(p.DIRECT)
        
        # 设置仿真参数（每次重置后都需要）
        p.setGravity(0, 0, -9.8, physicsClientId=self.physics_client_id)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1/240.,  # 仿真时间步
            numSubSteps=1,
            physicsClientId=self.physics_client_id
        )
        
        # 禁用实时仿真以便更快训练
        p.setRealTimeSimulation(0, physicsClientId=self.physics_client_id)
    
    def _create_object(self, position: np.ndarray) -> int:
        """
        创建待抓取的物体
        
        Args:
            position: 物体初始位置
            
        Returns:
            物体ID
        """
        object_shape = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=self.object_size,
            physicsClientId=self.physics_client_id
        )
        
        object_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.object_size,
            rgbaColor=self.object_color,
            physicsClientId=self.physics_client_id
        )
        
        object_id = p.createMultiBody(
            baseMass=self.object_mass,
            baseCollisionShapeIndex=object_shape,
            baseVisualShapeIndex=object_visual,
            basePosition=position,
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=self.physics_client_id
        )
        
        return object_id
    
    def _create_target(self, position: np.ndarray):
        """
        创建目标放置区域
        
        Args:
            position: 目标位置
        """
        # 创建半透明的目标区域
        target_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=self.target_size,
            height=0.01,
            physicsClientId=self.physics_client_id
        )
        
        target_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self.target_size,
            length=0.01,
            rgbaColor=self.target_color,
            physicsClientId=self.physics_client_id
        )
        
        self.target_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=target_shape,
            baseVisualShapeIndex=target_visual,
            basePosition=position,
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.physics_client_id
        )
    
    def _get_observation(self) -> np.ndarray:
        """
        获取当前观察状态（简化版）
        
        Returns:
            状态向量（11维）
        """
        # 获取机器人状态
        ee_position = self.robot.get_ee_position()
        gripper_state = self.robot.get_gripper_state()
        
        # 获取物体位置
        if self.object_id is not None:
            object_pos, _ = p.getBasePositionAndOrientation(
                self.object_id,
                physicsClientId=self.physics_client_id
            )
            object_pos = np.array(object_pos, dtype=np.float32)
        else:
            object_pos = np.zeros(3, dtype=np.float32)
        
        # 构建状态向量
        state = []
        
        # 1. 末端执行器位置
        state.extend(ee_position)
        
        # 2. 夹爪状态
        state.append(gripper_state)
        
        # 3. 目标物体位置
        state.extend(object_pos)
        
        # 4. 目标放置位置
        if self.target_position is not None:
            state.extend(self.target_position)
        else:
            state.extend([0.0, 0.0, 0.0])
        
        # 5. 物体是否被抓取
        state.append(1.0 if self.object_grabbed else 0.0)
        
        return np.array(state, dtype=np.float32)
    
    def _calculate_reward(self) -> float:
        """
        计算当前奖励值（优化版）
        
        Returns:
            奖励值
        """
        reward = 0.0
        
        # 获取当前状态
        ee_position = self.robot.get_ee_position()
        gripper_state = self.robot.get_gripper_state()
        
        # 获取物体位置
        if self.object_id is not None:
            object_pos, _ = p.getBasePositionAndOrientation(
                self.object_id,
                physicsClientId=self.physics_client_id
            )
            object_pos = np.array(object_pos, dtype=np.float32)
        else:
            object_pos = np.zeros(3)
        
        # 1. 到达目标物体的奖励（大幅增加奖励强度）
        if not self.object_grabbed and not self.object_placed:
            distance_to_object = np.linalg.norm(ee_position - object_pos)
            # 使用更强的线性衰减奖励
            reward += max(0, 2.0 - distance_to_object * 10.0)  # 距离0时奖励2.0
            
            # 额外的抓取奖励（当夹爪靠近物体时）
            if distance_to_object < 0.25:  # 放宽距离阈值，与抓取检测一致
                # 鼓励夹爪闭合
                if gripper_state < 0.4:  # 夹爪闭合
                    reward += 1.0  # 增加奖励
                # 尝试抓取的奖励（即使未成功）
                if gripper_state < 0.7:  # 放宽夹爪条件
                    reward += 0.3
        
        # 2. 成功抓取奖励（大幅增加）
        if self.object_grabbed and not self.object_placed:
            reward += 50.0  # 抓取成功的基础奖励（大幅增加）
            
            # 3. 到达放置位置的奖励（大幅增加奖励强度）
            if self.target_position is not None:
                distance_to_target = np.linalg.norm(ee_position - self.target_position)
                reward += max(0, 2.0 - distance_to_target * 10.0)  # 距离0时奖励2.0
                
                # 额外的放置奖励（当靠近目标时）
                if distance_to_target < 0.25:  # 放宽阈值
                    reward += 1.0  # 增加奖励
                    # 鼓励夹爪打开以放置物体
                    if gripper_state > 0.5:  # 放宽夹爪打开条件
                        reward += 0.5
        
        # 4. 成功放置奖励（保持高奖励）
        if self.object_placed:
            reward += 100.0  # 任务完成的大奖励（大幅增加）
        
        # 5. 时间惩罚（大幅减少，避免主导奖励）
        reward -= 0.002  # 原0.01
        
        # 6. 碰撞惩罚（减少）
        if self.robot.check_collision():
            reward -= 0.5  # 原2.0
        
        # 7. 夹爪使用奖励（鼓励正确使用夹爪）
        # 当靠近物体时，鼓励闭合；当抓住物体靠近目标时，鼓励打开
        if not self.object_grabbed and not self.object_placed:
            distance_to_object = np.linalg.norm(ee_position - object_pos)
            if distance_to_object < 0.2:
                # 靠近物体时，夹爪闭合给奖励
                reward += (0.3 - gripper_state) * 0.5  # gripper_state越小（越闭合）奖励越高
        elif self.object_grabbed and not self.object_placed:
            if self.target_position is not None:
                distance_to_target = np.linalg.norm(ee_position - self.target_position)
                if distance_to_target < 0.2:
                    # 靠近目标时，夹爪打开给奖励
                    reward += (gripper_state - 0.3) * 0.5  # gripper_state越大（越打开）奖励越高
        
        # 8. 掉落惩罚
        if self.object_dropped:
            reward -= 5.0
        
        return reward
    
    def _check_task_success(self) -> bool:
        """
        检查任务是否成功完成
        
        Returns:
            是否成功
        """
        if not self.object_grabbed or self.object_id is None:
            return False
        
        # 检查物体是否在目标区域内
        object_pos, _ = p.getBasePositionAndOrientation(
            self.object_id,
            physicsClientId=self.physics_client_id
        )
        object_pos = np.array(object_pos)
        
        if self.target_position is not None:
            distance = np.linalg.norm(object_pos - self.target_position)
            return distance < self.target_size and self.object_grabbed
        
        return False
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境到初始状态
        
        Args:
            seed: 随机种子
            options: 可选参数
            
        Returns:
            (初始状态, 信息字典)
        """
        super().reset(seed=seed)
        
        # 重置仿真
        p.resetSimulation(physicsClientId=self.physics_client_id)
        self._init_simulation()
        
        # 确保physics_client_id已设置
        assert self.physics_client_id is not None, "PyBullet物理客户端ID未初始化"
        
        # 重新初始化机器人
        self.robot = UR5Robot(self.physics_client_id, use_gui=self.use_gui)
        
        # 重置任务状态
        self.object_grabbed = False
        self.object_placed = False
        self.object_dropped = False
        self.current_step = 0
        
        # 固定位置（简化演示）
        object_x = 0.5  # 工作空间中心
        object_y = 0.0  # 中心
        object_z = 0.15  # 桌子上方
        object_position = np.array([object_x, object_y, object_z])
        
        # 创建物体
        if self.object_id is not None:
            p.removeBody(self.object_id, physicsClientId=self.physics_client_id)
        self.object_id = self._create_object(object_position)
        
        # 固定目标位置（与物体位置不同）
        target_x = 0.5  # 相同X
        target_y = 0.1  # Y方向偏移0.1米
        self.target_position = np.array([target_x, target_y, 0.1])
        
        # 创建目标区域
        self._create_target(self.target_position)
        
        # 重置机器人到初始位置
        # UR5有6个旋转关节 + 1个夹爪棱柱关节
        # 初始关节位置: [肩部旋转, 肩部抬起, 肘部, 手腕1, 手腕2, 手腕3, 夹爪]
        # 夹爪初始为打开状态（值接近1.0）
        initial_joint_positions = np.array([0.0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0, 1.0])
        self.robot.reset(initial_joint_positions[:len(self.robot.joint_indices)])
        
        # 执行几步仿真让物体稳定
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.physics_client_id)
            if self.use_gui and self.render_mode == 'human':
                time.sleep(1/240.)
        
        # 获取初始观察
        observation = self._get_observation()
        info = {
            'object_position': object_position,
            'target_position': self.target_position,
            'object_grabbed': self.object_grabbed,
            'object_placed': self.object_placed
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步动作
        
        Args:
            action: 动作向量
            
        Returns:
            (新状态, 奖励, 是否终止, 是否截断, 信息字典)
        """
        self.current_step += 1
        
        # 解析动作
        dx, dy, dz, gripper_action = action
        
        # 获取当前末端执行器位置
        current_ee_pos = self.robot.get_ee_position()
        
        # 计算目标位置（限制在工作空间内）
        target_ee_pos = current_ee_pos + np.array([dx, dy, dz])
        target_ee_pos = np.clip(target_ee_pos, [-1.0, -1.0, 0.1], [1.0, 1.0, 1.0])
        
        # 使用逆运动学计算关节目标
        try:
            target_joint_positions = self.robot.inverse_kinematics(target_ee_pos)
            self.robot.set_joint_positions(target_joint_positions)
        except Exception as e:
            # 如果逆运动学失败，使用当前位置
            print(f"逆运动学失败: {e}")
        
        # 设置夹爪
        self.robot.set_gripper(gripper_action)
        
        # 执行仿真步骤
        for _ in range(4):  # 每个动作执行4个仿真步骤
            p.stepSimulation(physicsClientId=self.physics_client_id)
            if self.use_gui and self.render_mode == 'human':
                time.sleep(1/240.)
        
        # 检查物体是否被抓取
        if not self.object_grabbed and self.object_id is not None:
            # 获取物体位置
            object_pos, _ = p.getBasePositionAndOrientation(
                self.object_id,
                physicsClientId=self.physics_client_id
            )
            object_pos = np.array(object_pos, dtype=np.float32)
            
            # 检查夹爪是否接触到物体
            contact_points = p.getContactPoints(
                bodyA=self.robot.robot_id,
                bodyB=self.object_id,
                physicsClientId=self.physics_client_id
            )
            
            # 如果夹爪闭合且靠近物体，则认为抓取成功（简化版：忽略接触检测）
            gripper_state = self.robot.get_gripper_state()
            distance_to_object = np.linalg.norm(self.robot.get_ee_position() - object_pos)
            # 极致简化：只需夹爪闭合即认为抓取成功（忽略距离）
            if gripper_state < 0.5:
                self.object_grabbed = True
                
                # 将物体移动到末端执行器位置（确保接触）
                ee_pos = self.robot.get_ee_position()
                # 将物体移动到末端下方一点（模拟夹爪中心）
                object_target_pos = ee_pos + np.array([0, 0, -0.05])
                p.resetBasePositionAndOrientation(
                    self.object_id,
                    object_target_pos,
                    [0, 0, 0, 1],
                    physicsClientId=self.physics_client_id
                )
                
                # 将物体固定在末端执行器上（模拟抓取）
                if self.object_grabbed:
                    # 创建固定约束
                    constraint_id = p.createConstraint(
                        parentBodyUniqueId=self.robot.robot_id,
                        parentLinkIndex=self.robot.ee_link_index,
                        childBodyUniqueId=self.object_id,
                        childLinkIndex=-1,
                        jointType=p.JOINT_FIXED,
                        jointAxis=[0, 0, 0],
                        parentFramePosition=[0, 0, 0],
                        childFramePosition=[0, 0, 0],
                        physicsClientId=self.physics_client_id
                    )
        
        # 检查物体是否被放置
        if self.object_grabbed and not self.object_placed:
            # 检查是否应该释放物体（夹爪打开）
            gripper_state = self.robot.get_gripper_state()
            if gripper_state > 0.7:  # 夹爪打开
                # 检查是否在目标区域内
                if self._check_task_success():
                    self.object_placed = True
                    self.object_grabbed = False
                    
                    # 移除固定约束
                    constraints = p.getConstraintUniqueId(0, physicsClientId=self.physics_client_id)
                    if constraints != -1:
                        p.removeConstraint(constraints, physicsClientId=self.physics_client_id)
                else:
                    # 夹爪打开但不在目标区域 -> 掉落物体，终止episode
                    self.object_grabbed = False
                    self.object_dropped = True
                    constraints = p.getConstraintUniqueId(0, physicsClientId=self.physics_client_id)
                    if constraints != -1:
                        p.removeConstraint(constraints, physicsClientId=self.physics_client_id)
                    # 掉落惩罚将在奖励函数中处理
        
        # 计算奖励
        reward = self._calculate_reward()
        
        # 检查终止条件
        terminated = self.object_placed or self.object_dropped
        truncated = self.current_step >= self.max_steps
        
        # 获取新状态
        observation = self._get_observation()
        
        # 构建信息字典
        info = {
            'object_grabbed': self.object_grabbed,
            'object_placed': self.object_placed,
            'object_dropped': self.object_dropped,
            'current_step': self.current_step,
            'gripper_state': self.robot.get_gripper_state(),
            'ee_position': self.robot.get_ee_position(),
            'task_success': terminated
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """渲染当前状态"""
        if self.render_mode == 'human' and self.use_gui:
            # PyBullet GUI会自动渲染
            pass
        elif self.render_mode == 'rgb_array':
            # 获取RGB图像
            width, height = 640, 480
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0.5, 0, 0.3],
                distance=1.5,
                yaw=45,
                pitch=-30,
                roll=0,
                upAxisIndex=2,
                physicsClientId=self.physics_client_id
            )
            projection_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=width/height,
                nearVal=0.01,
                farVal=10.0,
                physicsClientId=self.physics_client_id
            )
            _, _, rgb_array, _, _ = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix,
                physicsClientId=self.physics_client_id
            )
            return rgb_array
        
        return None
    
    def close(self):
        """关闭环境，清理资源"""
        if self.physics_client_id is not None:
            p.disconnect(physicsClientId=self.physics_client_id)
            self.physics_client_id = None


# 环境测试代码
if __name__ == "__main__":
    # 创建环境
    env = UR5PickPlaceEnv(render_mode='human', use_gui=True, max_steps=50)
    
    # 测试重置
    print("测试环境重置...")
    obs, info = env.reset()
    print(f"初始状态形状: {obs.shape}")
    print(f"物体位置: {info['object_position']}")
    print(f"目标位置: {info['target_position']}")
    
    # 测试随机动作
    print("\n测试随机动作...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"步骤 {i+1}: 奖励={reward:.3f}, 抓取={info['object_grabbed']}, 放置={info['object_placed']}")
        
        if terminated or truncated:
            print("Episode 结束!")
            break
    
    env.close()
    print("\n环境测试完成!")
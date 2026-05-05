import numpy as np
from omegaconf import DictConfig
import random
from pathlib import Path

from utils.rich_print import log, print_info, print_error, print_warn, print_debug
from utils.terminal_monitor import TerminalMonitor
from .Robot import Robot

class Env:
    map_height: int
    """地图高度(行数)"""
    map_width: int
    """地图宽度(列数)"""
    map_depth: int
    """地图深度(层数)"""
    map_size: tuple
    """地图尺寸(行数, 列数, 层数)"""
    UUV: Robot | None
    """我方机器人"""
    enemy: Robot | None
    """敌方机器人"""

    def __init__(self, cfg: DictConfig) -> None:
        """环境类，核心训练场景

        :param cfg: 配置对象，包含环境参数
        :param console: rich Console 对象，用于打印日志
        """
        self.cfg = cfg

        self.map_width = cfg.env.map_width
        self.map_height = cfg.env.map_height
        self.map_depth = cfg.env.map_depth
        self.map_size = (self.map_height, self.map_width, self.map_depth)

        print_info(f"环境初始化完成: 地图尺寸 {self.map_size}")

        self.uuv = None
        self.enemy = None

        self.cumulative_acoustic_signal = 0
        self.reward = 0
        self.done = False
        self.terminate = False
        self.info = {}

        # 被动声呐探测方程 SL-TL-(NL-DI) > DT
        # SL: 声源声压级，单位dB
        self.SL = cfg.env.uuv_SL
        # TL: 传播损失，单位dB
        # NL: 环境噪声级，单位dB
        self.NL = cfg.env.NL
        # DI: 方向性增益，单位dB
        self.DI = cfg.env.DI
        # DT: 探测阈值，单位dB
        self.DT = cfg.env.DT

        # 读取地形信息 NPZ 文件
        npz_file = Path('output/bty/terrain.npz')
        data = np.load(npz_file)
        # 获取3D可通行性布尔数组，True表示山体（不可通行），False表示水/空隙（可通行）
        self.terrain_3d = data['terrain_3d']  # shape: (101, 101, 11), dtype: bool
        # 获取2D海深数组
        self.bathymetry_2d = data['bathymetry_2d']  # shape: (101, 101), dtype: float64
        print_info(f"地形数据加载完成: 3D可通行性数组形状 {self.terrain_3d.shape}, 2D海深数组形状 {self.bathymetry_2d.shape}")

    def reset(self):
        """重置环境，初始化机器人位置和状态"""

        # 从配置中获取米数单位的范围，转换为网格索引
        # 采样步长: X/Y=100米, Z=50米
        uuv_start_x_min_idx = int(self.cfg.env.uuv_start_x_min // 100)
        uuv_start_x_max_idx = int(self.cfg.env.uuv_start_x_max // 100)
        uuv_start_y_min_idx = int(self.cfg.env.uuv_start_y_min // 100)
        uuv_start_y_max_idx = int(self.cfg.env.uuv_start_y_max // 100)
        uuv_start_z_min_idx = int(self.cfg.env.uuv_start_z_min // 50)
        uuv_start_z_max_idx = int(self.cfg.env.uuv_start_z_max // 50)

        # 随机生成我方机器人初始位置，确保在可通行区域（使用网格索引）
        while True:
            temp_UUV_x = random.randint(uuv_start_x_min_idx, uuv_start_x_max_idx)
            temp_UUV_y = random.randint(uuv_start_y_min_idx, uuv_start_y_max_idx)
            temp_UUV_z = random.randint(uuv_start_z_min_idx, uuv_start_z_max_idx)

            if (self.terrain_3d[temp_UUV_y, temp_UUV_x, temp_UUV_z] == False):
                break
        self.uuv = Robot(temp_UUV_x, temp_UUV_y, temp_UUV_z)

        # 敌方位置也需要转换为网格索引
        enemy_x_idx = int(self.cfg.env.enemy_x // 100)
        enemy_z_idx = int(self.cfg.env.enemy_z // 50)
        enemy_y_min_idx = int(self.cfg.env.enemy_y_min // 100)
        enemy_y_max_idx = int(self.cfg.env.enemy_y_max // 100)

        # 随机生成敌方机器人初始位置，确保在可通行区域（使用网格索引）
        while True:
            temp_enemy_x = enemy_x_idx
            temp_enemy_y = random.randint(enemy_y_min_idx, enemy_y_max_idx)
            temp_enemy_z = enemy_z_idx

            if (self.terrain_3d[temp_enemy_y, temp_enemy_x, temp_enemy_z] == False):
                break
        self.enemy = Robot(temp_enemy_x, temp_enemy_y, temp_enemy_z)

        self.cumulative_acoustic_signal = 0
        self.reward = 0
        self.done = False
        self.terminate = False
        self.info = {}
        self.now_step = 0
        # 计算推荐的最短路径步数（曼哈顿距离 * 缩放因子）
        self.recommended_steps = (abs(self.uuv.x - self.enemy.x) + abs(self.uuv.y - self.enemy.y) + abs(self.uuv.z - self.enemy.z)) * self.cfg.env.max_recommand_step_scaling_factor

        print_info(f"环境重置完成: 我方机器人初始位置 ({self.uuv.x}, {self.uuv.y}, {self.uuv.z}), 敌方机器人初始位置 ({self.enemy.x}, {self.enemy.y}, {self.enemy.z})")

    def step(self, action):
        """执行一步环境交互，根据动作更新状态并计算奖励

        :param action: 机器人执行的动作，整数编码
        :return: 新状态、奖励、是否结束、额外信息
        """
        if self.uuv is None or self.enemy is None:
            raise RuntimeError("Environment is not reset. Please call reset() before step().")

        # 1.根据动作更新我方机器人位置
        self._move_robot(action, self.uuv, self.terrain_3d)
        self.now_step += 1

        # 2.查询计算声呐信号强度和奖励
        now_TL = self._query_TL(self.uuv, self.enemy)

        # 3.利用被动声呐探测方程 SL - TL - (NL - DI) > DT 计算当前对方处的接收声信号强度
        self.cumulative_acoustic_signal += self.SL - now_TL - (self.NL - self.DI)
        if (self.cumulative_acoustic_signal > self.DT * 2.2):  # 累积声呐信号强度超过探测阈值的2.2倍，认为被发现
            self.done = True
            self.reward = -100
            self.info['result'] = '被发现, 累计声信号强度: {:.2f} dB'.format(self.cumulative_acoustic_signal)
        elif (self.now_step >= self.recommended_steps * 2.5):  # 超过推荐最短路径步数的2.5倍，认为步数过多死亡
            self.done = True
            self.reward = -50
            self.info['result'] = '步数过多死亡'
        elif (self.uuv.x == 2000):
            self.done = True
            self.reward = 100
            self.info['result'] = '成功突防, 累计声信号强度: {:.2f} dB'.format(self.cumulative_acoustic_signal)

        # 4.若某次没发现，则清空累计声呐信号强度
        if (not self.done):
            self.cumulative_acoustic_signal = 0

        # 5.计算本步奖励，隐蔽性越高奖励越大，接近敌人有固定奖励，远离敌人有固定惩罚
        approach_reward = 0
        stealth_reward = 0
        # 计算隐蔽性奖励，传播损失越大奖励越高
        stealth_reward = -now_TL
        # 计算接近敌人奖励
        if (action == 0):
            approach_reward = 1 / self.recommended_steps

        self.reward = stealth_reward + approach_reward
        self.info['reward_details'] = {
            'stealth_reward': stealth_reward,
            'approach_reward': approach_reward
        }

        # 6.令敌人随机移动一步，敌人只能沿着y轴随机移动，且不能进入不可通行区域
        # self._enemy_step()

        return (self.uuv.x, self.uuv.y, self.uuv.z), self.reward, self.done, self.info

    def _move_robot(self, action, robot: Robot, terrain_3d):
        """根据动作更新机器人位置，考虑地形限制

        :param action: 动作编号，0-5分别对应六个方向
        :param robot: 机器人对象
        :param terrain_3d: 3D布尔数组，True表示不可通行，False表示可通行
        """
        # 定义动作对应的坐标变化
        action_map = {
            0: (-1, 0, 0),  # 向前
            1: (1, 0, 0),  # 向后
            2: (0, -1, 0),  # 向左
            3: (0, 1, 0),  # 向右
            4: (0, 0, -1),  # 向下
            5: (0, 0, 1),  # 向上
        }
        dx, dy, dz = action_map.get(action, (0, 0, 0))
        new_x = robot.x + dx
        new_y = robot.y + dy
        new_z = robot.z + dz

        # 检查新位置是否在地图范围内且可通行
        if (0 <= new_x < terrain_3d.shape[1] and 
            0 <= new_y < terrain_3d.shape[0] and 
            0 <= new_z < terrain_3d.shape[2] and 
            terrain_3d[new_y, new_x, new_z] == False):
            robot.x = new_x
            robot.y = new_y
            robot.z = new_z

    def _query_TL(self, uuv: Robot, enemy: Robot):
        """查询传播损失 TL，根据我方机器人和敌方机器人的位置计算

        :param uuv: 我方机器人对象
        :param enemy: 敌方机器人对象
        :return: 传播损失 TL，单位dB
        """
        # 计算我方机器人和敌方机器人的距离
        distance = np.sqrt((uuv.x - enemy.x) ** 2 + (uuv.y - enemy.y) ** 2 + (uuv.z - enemy.z) ** 2)
        # 根据距离计算传播损失 TL，假设传播损失与距离成正比
        TL = 20 * np.log10(distance + 1e-6)  # 加上一个小值避免log(0)
        return TL

    # def _enemy_step(self):
    #     """敌方机器人随机移动一步，沿y轴移动，考虑地形限制"""
    #     while True:
    #         self.enemy.y += random.choice([-1, 0, 1])
    #         if (self.enemy.y >= self.cfg.env.enemy_y_min and self.enemy.y <= self.cfg.env.enemy_y_max and self.terrain_3d[self.enemy.y, self.enemy.x, self.enemy.z] == False):
    #             break

    def get_observation_tensor(self, device: str = "cuda", window_size: int = 16):
        """为 ACNet 生成观测张量（spatial_input 与 state_vector）。

        功能说明:
            生成 ACNet 所需的两路输入张量，所有计算直接在 GPU 执行：
            1. spatial_input: (1, 2, D, H, W) 的 float32 张量
               - 通道0: 传播损失 TL 热力图（围绕 UUV 的局部窗口）
               - 通道1: 地形可通行性（True=不可通行, False=可通行，1=True, 0=False）
            2. state_vector: (1, 6) 的 float32 张量
               - (x_uuv, y_uuv, z_uuv, x_enemy, y_enemy, z_enemy) 的绝对坐标
            
            坐标约定: 地形索引为 terrain_3d[y, x, z]，向量存储为 (x, y, z)。
            张量设备: 所有张量直接在指定设备（默认 cuda）上创建。

        输入参数:
            device (str): 计算设备，默认为 "cuda"。
            window_size (int): spatial 窗口半径（单位：网格单元数），默认为 16。
                最终 spatial_input 形状为 (1, 2, window_size, window_size, window_size)。

        输出参数:
            Tuple[torch.Tensor, torch.Tensor]:
                spatial_input: 形状 (1, 2, window_size, window_size, window_size)，dtype=float32，device=指定设备。
                state_vector: 形状 (1, 6)，dtype=float32，device=指定设备。

        调用示例:
            >>> spatial_input, state_vector = env.get_observation_tensor(device="cuda", window_size=16)
            >>> spatial_input.shape, state_vector.shape
            (torch.Size([1, 2, 16, 16, 16]), torch.Size([1, 6]))
        """
        import torch

        if self.uuv is None or self.enemy is None:
            raise RuntimeError("环境尚未重置。请先调用 reset() 再获取观测张量。")

        # --- 构建 spatial_input ---
        # 初始化两通道的 spatial 特征数组（CPU numpy，后续转 GPU torch）
        spatial_array = np.zeros((2, window_size, window_size, window_size), dtype=np.float32)

        # 计算窗口中心（UUV 位置）与边界范围
        center_x, center_y, center_z = self.uuv.x, self.uuv.y, self.uuv.z
        z_min = max(0, center_z - window_size // 2)
        z_max = min(self.map_depth, center_z + window_size // 2)
        y_min = max(0, center_y - window_size // 2)
        y_max = min(self.map_height, center_y + window_size // 2)
        x_min = max(0, center_x - window_size // 2)
        x_max = min(self.map_width, center_x + window_size // 2)

        # 窗口在 spatial_array 内的起始偏移
        z_offset = window_size // 2 - (center_z - z_min)
        y_offset = window_size // 2 - (center_y - y_min)
        x_offset = window_size // 2 - (center_x - x_min)

        # 通道0: TL 热力图（局部窗口内每个点到敌方的传播损失）
        for z_idx in range(z_min, z_max):
            for y_idx in range(y_min, y_max):
                for x_idx in range(x_min, x_max):
                    # 计算该点到敌方的距离（以及 TL）
                    distance = np.sqrt(
                        (x_idx - self.enemy.x) ** 2
                        + (y_idx - self.enemy.y) ** 2
                        + (z_idx - self.enemy.z) ** 2
                    )
                    tl_value = 20 * np.log10(distance + 1e-6)
                    
                    # 映射到 spatial_array 坐标
                    sa_z = z_offset + (z_idx - z_min)
                    sa_y = y_offset + (y_idx - y_min)
                    sa_x = x_offset + (x_idx - x_min)
                    
                    if 0 <= sa_z < window_size and 0 <= sa_y < window_size and 0 <= sa_x < window_size:
                        spatial_array[0, sa_z, sa_y, sa_x] = tl_value

        # 通道1: 地形可通行性（1=不可通行, 0=可通行）
        for z_idx in range(z_min, z_max):
            for y_idx in range(y_min, y_max):
                for x_idx in range(x_min, x_max):
                    # 地形索引约定: terrain_3d[y, x, z]
                    passable = 1.0 if self.terrain_3d[y_idx, x_idx, z_idx] else 0.0
                    
                    sa_z = z_offset + (z_idx - z_min)
                    sa_y = y_offset + (y_idx - y_min)
                    sa_x = x_offset + (x_idx - x_min)
                    
                    if 0 <= sa_z < window_size and 0 <= sa_y < window_size and 0 <= sa_x < window_size:
                        spatial_array[1, sa_z, sa_y, sa_x] = passable

        # 转换为 PyTorch 张量，并移至目标设备
        spatial_input = torch.from_numpy(spatial_array).unsqueeze(0).to(device)  # (1, 2, D, H, W)

        # --- 构建 state_vector ---
        # 状态向量: (x_uuv, y_uuv, z_uuv, x_enemy, y_enemy, z_enemy)
        state_vector_np = np.array(
            [
                [
                    float(self.uuv.x),
                    float(self.uuv.y),
                    float(self.uuv.z),
                    float(self.enemy.x),
                    float(self.enemy.y),
                    float(self.enemy.z),
                ]
            ],
            dtype=np.float32,
        )
        state_vector = torch.from_numpy(state_vector_np).to(device)  # (1, 6)

        return spatial_input, state_vector
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

        # 随机生成我方机器人初始位置，确保在可通行区域
        while True:
            temp_UUV_x = random.randint(self.cfg.env.uuv_start_x_min, self.cfg.env.uuv_start_x_max)
            temp_UUV_y = random.randint(self.cfg.env.uuv_start_y_min, self.cfg.env.uuv_start_y_max)
            temp_UUV_z = random.randint(self.cfg.env.uuv_start_z_min, self.cfg.env.uuv_start_z_max)

            if (self.terrain_3d[temp_UUV_y, temp_UUV_x, temp_UUV_z] == False):
                break
        self.uuv = Robot(temp_UUV_x, temp_UUV_y, temp_UUV_z)

        # 随机生成敌方机器人初始位置，确保在可通行区域
        while True:
            temp_enemy_x = self.cfg.env.enemy_x
            temp_enemy_y = random.randint(self.cfg.env.enemy_y_min, self.cfg.env.enemy_y_max)
            temp_enemy_z = self.cfg.env.enemy_z

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
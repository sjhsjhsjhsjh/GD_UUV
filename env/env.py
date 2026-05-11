import numpy as np
from omegaconf import DictConfig
import random
from pathlib import Path
from torch import Tensor, from_numpy
from numpy import tanh

from utils.rich_print import log, print_info, print_error, print_warn, print_debug, print_success
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
    UUV: Robot
    """我方机器人"""
    enemy: Robot
    """敌方机器人"""
    terrain_3d: np.ndarray
    """3D可通性布尔转float32数组，1.0表示可通行，0.0表示不可通行，shape=(map_width, map_height, map_depth)"""

    def __init__(self, cfg: DictConfig) -> None:
        """环境类，核心训练场景

        :param cfg: 配置对象，包含环境参数
        :param console: rich Console 对象，用于打印日志
        """
        self.cfg = cfg

        self.map_width = cfg.env.map_width
        self.map_height = cfg.env.map_height
        self.map_depth = cfg.env.map_depth
        self.map_size = (self.map_width, self.map_height, self.map_depth)

        print_info(f"环境初始化完成: 地图尺寸 {self.map_size}")

        self.action_map = {
            0: (-1, 0, 0),  # 向前
            1: (1, 0, 0),  # 向后
            2: (0, -1, 0),  # 向左
            3: (0, 1, 0),  # 向右
            4: (0, 0, -1),  # 向下
            5: (0, 0, 1),  # 向上
            6: (0, 0, 0),  # 原地不动
        }
        # 从配置中获取米数单位的范围，转换为网格索引
        # 采样步长: X/Y=100米, Z=50米
        self.uuv_x_min_index = int(cfg.env.uuv_x_min // cfg.env.sampling_x_step)
        self.uuv_x_max_index = int(cfg.env.uuv_x_max // cfg.env.sampling_x_step)
        self.uuv_y_min_index = int(cfg.env.uuv_y_min // cfg.env.sampling_y_step)
        self.uuv_y_max_index = int(cfg.env.uuv_y_max // cfg.env.sampling_y_step)
        self.uuv_z_min_index = int(cfg.env.uuv_z_min // cfg.env.sampling_z_step)
        self.uuv_z_max_index = int(cfg.env.uuv_z_max // cfg.env.sampling_z_step)
        self.uuv_start_x_min_idx = int(self.cfg.env.uuv_start_x_min // self.cfg.env.sampling_x_step)
        self.uuv_start_x_max_idx = int(self.cfg.env.uuv_start_x_max // self.cfg.env.sampling_x_step)
        self.uuv_start_y_min_idx = int(self.cfg.env.uuv_start_y_min // self.cfg.env.sampling_y_step)
        self.uuv_start_y_max_idx = int(self.cfg.env.uuv_start_y_max // self.cfg.env.sampling_y_step)
        self.uuv_start_z_min_idx = int(self.cfg.env.uuv_start_z_min // self.cfg.env.sampling_z_step)
        self.uuv_start_z_max_idx = int(self.cfg.env.uuv_start_z_max // self.cfg.env.sampling_z_step)
        # 敌方位置也需要转换为网格索引
        self.enemy_x_idx = int(self.cfg.env.enemy_x // self.cfg.env.sampling_x_step)
        self.enemy_z_idx = int(self.cfg.env.enemy_z // self.cfg.env.sampling_z_step)
        self.enemy_y_min_idx = int(self.cfg.env.enemy_y_min // self.cfg.env.sampling_y_step)
        self.enemy_y_max_idx = int(self.cfg.env.enemy_y_max // self.cfg.env.sampling_y_step)

        self.field_of_view = self.cfg.env.field_of_view
        self.field_of_view_on_z = self.cfg.env.field_of_view_on_z
        self.victory_x_idx = int(self.cfg.env.victory_x // self.cfg.env.sampling_x_step)
        self.recommand_step_scaling_factor = self.cfg.env.recommand_step_scaling_factor
        self.max_recommand_step_scaling_factor = self.cfg.env.max_recommand_step_scaling_factor
        self.cumulative_acoustic_signal = 0
        self.reward = 0
        self.done = False
        self.terminate = False
        self.info = {}

        # 被动声呐探测方程 SL-TL-(NL-DI) > DT
        # SL: 135dB，NL: 85dB，DI: 28dB，DT: 12dB, 代入方程，NL-DI=85-28=57dB,
        # SL-TL-57 > 12 → TL < SL-57-12 = 66dB
        # 探测中心频率为 66dB。根据环境中的分布，我希望，
        # 当TL大于69dB时，认为此时绝对隐蔽，是隐蔽突击的绝佳点位。
        # 当TL处于63~69dB时，是无海山遮挡下典型的传播损失，此时希望UUV能快速通过或者进行机动，尝试前往下一个隐蔽位置。
        # 当TL小于63dB时，认为此时绝对危险，不应该处于此环境下大于3个step。
        # SL: 声源声压级，单位dB
        self.SL = cfg.env.uuv_SL
        # TL: 传播损失，单位dB
        # NL: 环境噪声级，单位dB
        self.NL = cfg.env.NL
        # DI: 方向性增益，单位dB
        self.DI = cfg.env.DI
        # DT: 探测阈值，单位dB
        self.DT = cfg.env.DT
        # 根据声呐探测方程计算探测中心频率 TL，作为环境中隐蔽奖励的参考值
        self.tl_proper = self.SL - self.NL + self.DI - self.DT
        # TL 容差
        self.tl_tolerance = cfg.env.tl_tolerance
        print_info(f"被动声呐探测方程参数: SL={self.SL}dB, NL={self.NL}dB, DI={self.DI}dB, DT={self.DT}dB → 探测中心频率 TL = {self.tl_proper}dB, TL容差 = {self.tl_tolerance}dB")

        # 读取地形信息 NPZ 文件
        npz_file = Path('output/bty/terrain.npz')
        data = np.load(npz_file)
        # 获取3D可通行性布尔数组，True表示山体（不可通行），False表示水/空隙（可通行）
        self.terrain_3d = data['terrain_3d']  # shape: (101, 101, 11), dtype: bool
        self.terrain_3d = np.transpose(self.terrain_3d, axes=(1, 0, 2))  # 转换为 (x, y, z) 顺序，shape: (101, 101, 11)
        # 转换为 float32，山体=0.0，水/空隙=1.0，适配 Relu 激活函数
        self.terrain_3d = np.logical_not(self.terrain_3d)  # 取反，使得 True（山体）变为 False，False（水/空隙）变为 True
        self.terrain_3d = self.terrain_3d.astype(np.float32)
        # 从现在开始，self.terrain_3d[x, y, z] 的访问方式表示地图坐标 (x, y, z) 处的地形信息，1.0 表示可通行，0.0 表示山体不可通行
        # 将 配置文件设置的可通行范围外的地形全部设为不可通行
        x_idx = np.arange(self.terrain_3d.shape[0])[:, None, None]
        y_idx = np.arange(self.terrain_3d.shape[1])[None, :, None]
        z_idx = np.arange(self.terrain_3d.shape[2])[None, None, :]

        outside_mask = (
            (x_idx < self.uuv_x_min_index) | (x_idx > self.uuv_x_max_index) |
            (y_idx < self.uuv_y_min_index) | (y_idx > self.uuv_y_max_index) |
            (z_idx < self.uuv_z_min_index) | (z_idx > self.uuv_z_max_index)
        )

        self.terrain_3d[outside_mask] = 0.0
        # 获取2D海深数组
        self.bathymetry_2d = data['bathymetry_2d']  # shape: (101, 101), dtype: float64
        self.bathymetry_2d = self.bathymetry_2d.astype(np.float32)
        self.bathymetry_2d = np.transpose(self.bathymetry_2d, axes=(1, 0))  # 转换为 (x, y) 顺序，shape: (101, 101)
        print_success(f"地形数据加载完成: 3D可通行性数组形状 {self.terrain_3d.shape}, 2D海深数组形状 {self.bathymetry_2d.shape}")

        # 加载TL信息文件
        temp_ret = self._load_tl_from_txt(cfg.env.tl_table_path)
        if temp_ret is None:
            print_error("TL信息加载失败")
            return

        # 加载地形数据至 GPU
        import torch
        self.terrain_3d_tensor = from_numpy(np.ascontiguousarray(self.terrain_3d)).to(device="cuda", dtype=torch.float32)
        self.tl_table_tensor = from_numpy(np.ascontiguousarray(self.tl_table)).to(device="cuda", dtype=torch.float32)
        # 在GPU上预分配网络观测的数组
        self.net_obervation_spatial_tensor = torch.zeros((2, self.field_of_view, self.field_of_view, self.field_of_view_on_z), dtype=torch.float32, device="cuda")
        # 直接使用配置中的 state_vector 维度（信任配置）并在 GPU 上预分配
        self.state_vector_dim = int(cfg.ppo.state_vector_dim)
        print_info(f"当前 state_vector 维度为：{self.state_vector_dim}")
        self.net_state_vector_tensor = torch.zeros((1, self.state_vector_dim), dtype=torch.float32, device="cuda")

        self._test_save_tl_slices_as_images()
        self._test_terrain_slices_as_images()

    def _test_terrain_slices_as_images(self):
        """测试：保存地形切片图像，验证坐标对应关系"""
        import matplotlib.pyplot as plt
        from pathlib import Path

        output_dir = Path("output/terrain_slices")
        output_dir.mkdir(parents=True, exist_ok=True)

        p1_x = 79
        p1_y = 31
        p1_z = 2
        print_info("terrain_3d[{}, {}, {}] = {}".format(p1_x, p1_y, p1_z, self.terrain_3d[p1_x, p1_y, p1_z]))
        p2_x = 87
        p2_y = 48
        p2_z = 4
        print_info("terrain_3d[{}, {}, {}] = {}".format(p2_x, p2_y, p2_z, self.terrain_3d[p2_x, p2_y, p2_z]))

        x = np.arange(self.terrain_3d.shape[0])
        y = np.arange(self.terrain_3d.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")

        for z in range(self.map_depth):
            terrain_slice = self.terrain_3d[:, :, z]
            plt.figure(figsize=(6, 6))
            plt.pcolormesh(X, Y, self.terrain_3d[:, :, z], shading='auto')
            plt.xlabel("X Coordinate")
            plt.ylabel("Y Coordinate")
            plt.colorbar(label="Passability (1=passable, 0=mountain)")
            plt.title(f"Terrain Slice at z={z} (Depth {z * self.cfg.env.sampling_z_step}m)")
            plt.xlabel("X Coordinate (grid index)")
            plt.ylabel("Y Coordinate (grid index)")
            # 画出 P1 和 P2 的位置，标记具有一定透明度
            if z == p1_z:
                plt.scatter([p1_x], [p1_y], color="red", marker="o", label="P1 (79, 31, 2)", alpha=0.5)
            if z == p2_z:
                plt.scatter([p2_x], [p2_y], color="blue", marker="x", label="P2 (87, 48, 4)", alpha=0.5)
            plt.legend()
            plt.savefig(output_dir / f"terrain_slice_z{z}.png", bbox_inches="tight", dpi=150)
            plt.close()

        print_success(f"地形切片图像保存完成: {output_dir.resolve()}")

    def _test_save_tl_slices_as_images(self):
        """测试：保存 TL 切片并将地形不可通行区域叠加为半透明红色图层。"""
        import matplotlib.pyplot as plt
        from pathlib import Path
        import numpy as np

        output_dir = Path("output/tl_slices")
        output_dir.mkdir(parents=True, exist_ok=True)

        enemy_y = 50  # 固定敌方 y 坐标，测试不同 z 层的 TL 切片

        x = np.arange(self.terrain_3d.shape[0])
        y = np.arange(self.terrain_3d.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")

        for z in range(1, 6):
            # TL: 索引顺序 [enemy_y, uuv_x, uuv_y, uuv_z]
            tl_slice = self.tl_table[enemy_y, :, :, z].astype(np.float32)

            terrain_slice = self.terrain_3d[:, :, z]
            # passable: True 表示可通行（值大于 0.5），mountain_mask 表示不可通行的点
            mountain_mask = (terrain_slice <= 0.5) 

            plt.figure(figsize=(6, 6))
            plt.pcolormesh(X, Y, tl_slice, shading='auto', cmap="viridis")
            plt.colorbar(label="TL (dB)")
            plt.title(f"TL Slice at z={z} (Depth {z * self.cfg.env.sampling_z_step}m)")
            plt.xlabel("X Coordinate (grid index)")
            plt.ylabel("Y Coordinate (grid index)")
            plt.savefig(output_dir / f"tl_slice_z{z}_plain.png", bbox_inches="tight", dpi=150)

            # 叠加不可通行区域：只显示山体部分，半透明绿色
            # masked = np.ma.masked_where(~mountain_mask, mountain_mask)
            # plt.pcolormesh(X, Y, masked, cmap="Greens", alpha=0.5, origin="lower", interpolation="nearest")

            plt.title(f"TL Slice at z={z} (Depth {z * self.cfg.env.sampling_z_step}m)")
            plt.xlabel("X Coordinate (grid index)")
            plt.ylabel("Y Coordinate (grid index)")
            plt.savefig(output_dir / f"tl_slice_z{z}.png", bbox_inches="tight", dpi=150)
            plt.close()

        print_success(f"TL切片图像保存完成: {output_dir.resolve()}")

    def reset(self):
        """重置环境，初始化机器人位置和状态"""

        # 随机生成我方机器人初始位置，确保在可通行区域（使用网格索引）
        while True:
            temp_UUV_x = random.randint(self.uuv_start_x_min_idx, self.uuv_start_x_max_idx)
            temp_UUV_y = random.randint(self.uuv_start_y_min_idx, self.uuv_start_y_max_idx)
            temp_UUV_z = random.randint(self.uuv_start_z_min_idx, self.uuv_start_z_max_idx)

            if (self.terrain_3d[temp_UUV_x, temp_UUV_y, temp_UUV_z] > 0.5):
                break
        self.uuv = Robot(temp_UUV_x, temp_UUV_y, temp_UUV_z)
        self.last_uuv_location = (self.uuv.x, self.uuv.y, self.uuv.z)
        self.uuv_start_x = self.uuv.x

        # 随机生成敌方机器人初始位置，确保在可通行区域（使用网格索引）
        while True:
            temp_enemy_x = self.enemy_x_idx
            temp_enemy_y = random.randint(self.enemy_y_min_idx, self.enemy_y_max_idx)
            temp_enemy_z = self.enemy_z_idx

            if (self.terrain_3d[temp_enemy_x, temp_enemy_y, temp_enemy_z] > 0.5):
                break
        self.enemy = Robot(temp_enemy_x, temp_enemy_y, temp_enemy_z)

        # 随机一个敌人初始移动方向，1表示沿y轴正方向，-1表示沿y轴负方向
        self.enemy_forward_direction = random.choice([-1, 1])

        self.cumulative_acoustic_signal = 0
        self.reward = 0
        self.done = False
        self.terminate = False
        self.info = {}
        self.now_step = 0
        temp = self._query_TL(self.enemy.y, self.uuv.x, self.uuv.y, self.uuv.z)
        self.上次TL = temp
        self.当前TL = temp
        # 计算推荐的最短路径步数（曼哈顿距离 * 缩放因子）
        self.manhattan_distance = abs(self.uuv.x - self.enemy.x) + abs(self.uuv.y - self.enemy.y) + abs(self.uuv.z - self.enemy.z)
        self.recommended_steps = self.manhattan_distance * self.recommand_step_scaling_factor
        self.recommended_max_steps = self.manhattan_distance * self.max_recommand_step_scaling_factor
        self.计算范围平均TL奖励() # 内部填充 self.区域平均TL

        # print_info(f"环境重置完成: 我方机器人初始位置 ({self.uuv.x}, {self.uuv.y}, {self.uuv.z}), 敌方机器人初始位置 ({self.enemy.x}, {self.enemy.y}, {self.enemy.z}), 推荐最短路径步数 {self.recommended_steps:.2f}")

    def step(self, action):
        """执行一步环境交互，根据动作更新状态并计算奖励

        :param action: 机器人执行的动作，整数编码
        :return: 新状态、奖励、是否结束、额外信息
        """

        # 1.根据动作更新我方机器人位置
        self.last_uuv_location = (self.uuv.x, self.uuv.y, self.uuv.z)
        self._move_robot(self.uuv, action)
        self.now_step += 1

        # 2.判断是否越界或者撞山，否则查TL表可能会直接爆了
        if (self.uuv.x < self.uuv_x_min_index or self.uuv.x > self.uuv_x_max_index or
            self.uuv.y < self.uuv_y_min_index or self.uuv.y > self.uuv_y_max_index or
            self.uuv.z < self.uuv_z_min_index or self.uuv.z > self.uuv_z_max_index):
            self.done = True
            self.reward = -10
            self.info['result'] = '超出边界, 位置: ({}, {}, {})'.format(self.uuv.x, self.uuv.y, self.uuv.z)
        # 撞山，立即死亡
        if (not self.done and self.terrain_3d[self.uuv.x, self.uuv.y, self.uuv.z] <= 0.5):
            self.done = True
            self.reward = -10
            self.info['result'] = '撞山, 位置: ({}, {}, {})'.format(self.uuv.x, self.uuv.y, self.uuv.z)

        # 仅在不越界、不撞山的情况下，才查询 TL 表（避免数组越界）
        if not self.done:
            # 3.查询计算声呐信号强度和奖励
            last_TL = self._query_TL(self.enemy.y, *self.last_uuv_location)
            now_TL = self._query_TL(self.enemy.y, self.uuv.x, self.uuv.y, self.uuv.z)
            self.上次TL = last_TL
            self.当前TL = now_TL
            if (now_TL < self.tl_proper - self.tl_tolerance):       # 当前点被发现，进入被发现状态，累计声呐信号强度
                self.cumulative_acoustic_signal += self.tl_proper - now_TL
            elif (now_TL > self.tl_proper + self.tl_tolerance):       # 当前点绝对隐蔽，进入绝对隐蔽状态，清空累计声呐信号强度
                self.cumulative_acoustic_signal = 0
            else:       # 当前点处于典型传播损失状态，敌方进入丢失状态，累计声呐信号强度减半
                self.cumulative_acoustic_signal *= 0.65
                self.cumulative_acoustic_signal += self.tl_proper - now_TL

            # 4. 查看其它死法，或者是赢了
            # 被发现，立即死亡
            if (self.cumulative_acoustic_signal > self.tl_tolerance * 2.2):  # 累积声呐信号强度超过探测阈值的2.2倍，认为被发现
                self.done = True
                self.reward = -3
                self.info['result'] = '被发现, 累计声信号强度: {:.2f} dB, 当前位置: ({}, {}, {}), 敌方当前位置: {}, 步数: {}, 当前TL: {:.2f}, 上次TL: {:.2f}'.format(self.cumulative_acoustic_signal, self.uuv.x, self.uuv.y, self.uuv.z, self.enemy.y, self.now_step, self.当前TL, self.上次TL)
            # 达到推荐步数上限，立即死亡
            if (not self.done and self.now_step > self.recommended_max_steps):
                self.done = True
                self.reward = -6
                self.info['result'] = '超出推荐步数上限, 当前步数: {}, 推荐最大步数: {:.2f}'.format(self.now_step, self.recommended_max_steps)
            # 达到敌人附近位置，立即胜利
            if (not self.done and self.uuv.x <= self.victory_x_idx):
                self.done = True
                self.reward = 20
                self.info['result'] = '胜利, 达到敌人附近位置: ({}, {}, {}), 消耗步数: {}, 与曼哈顿距离比例: {:.2f}'.format(self.uuv.x, self.uuv.y, self.uuv.z, self.now_step, self.now_step / self.manhattan_distance)

            # 4.计算本步奖励（仅当未触发死亡条件时）
            if not self.done:
                # 隐蔽性奖励
                隐蔽奖励 = self.计算隐蔽奖励(now_TL)
                # 靠近奖励
                靠近奖励 = self.计算靠近奖励(action)
                # TL 梯度奖励
                TL梯度奖励 = self.计算TL梯度奖励(last_TL, now_TL)
                # 范围平均 TL 奖励
                范围平均TL奖励 = self.计算范围平均TL奖励()
                # 固定时间惩罚
                固定时间惩罚 = - 1 / self.recommended_steps * 8

                self.reward = 隐蔽奖励 + TL梯度奖励 + 靠近奖励 + 范围平均TL奖励 + 固定时间惩罚
                self.info['reward_details'] = {
                    'stealth_reward': 隐蔽奖励,
                    'approach_reward': 靠近奖励,
                    'tl_gradient_reward': TL梯度奖励,
                    'area_average_tl_reward': 范围平均TL奖励
                }
            else:
                # 如果触发死亡条件，reward 已在条件分支中设置，只需设置空的 reward_details
                self.info['reward_details'] = {
                    'stealth_reward': 0,
                    'approach_reward': 0,
                    'tl_gradient_reward': 0,
                    'area_average_tl_reward': 0
                }

        # 5.令敌人随机移动一步，敌人只能沿着y轴随机移动，且不能进入不可通行区域
        self._enemy_step()

        return (self.uuv.x, self.uuv.y, self.uuv.z), self.reward, self.done, self.info

    def _load_tl_from_txt(self, tl_file_path: str):
        """从文本文件加载TL信息，构建一个字典以供查询。文本文件内容为 enemy_y,uuv_x,uuv_y,uuv_z,tl

        :param tl_file_path: TL信息文本文件路径，格式为每行 "enemy_y,uuv_x,uuv_y,uuv_z,TL_value"
        :return: TL信息字典，键为 (enemy_y, uuv_x, uuv_y, uuv_z) 坐标元组，值为对应的 TL 值
        """

        self.tl_table = np.zeros(shape=(101, 101, 101, 11))
        with open(tl_file_path, "r") as f:
            if f is None:
                print_error(f"无法打开TL信息文件: {tl_file_path}")
                return None
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 5:
                    enemy_y, uuv_x, uuv_y, uuv_z = map(int, parts[:4])
                    tl_value = float(parts[4])
                    self.tl_table[enemy_y, uuv_x, uuv_y, uuv_z] = tl_value

        # 需要调换 uuv_x 和 uuv_y 的位置，使得 tl_table[enemy_y, uuv_x, uuv_y, uuv_z] 的访问方式契合地图
        # self.tl_table = np.transpose(self.tl_table, axes=(0, 2, 1, 3))

        # 补充 TL 信息：计算范围内的地方，凡是不是山的位置，若TL信息为0，则补全为120dB
        for enemy_y in range(self.tl_table.shape[0]):
            for uuv_x in range(self.tl_table.shape[1]):
                for uuv_y in range(self.tl_table.shape[2]):
                    for uuv_z in range(self.tl_table.shape[3]):
                        if uuv_x <= self.uuv_x_max_index and uuv_x >= self.uuv_x_min_index \
                            and uuv_y <= self.uuv_y_max_index and uuv_y >= self.uuv_y_min_index \
                            and uuv_z <= self.uuv_z_max_index and uuv_z >= self.uuv_z_min_index \
                            and self.tl_table[enemy_y, uuv_x, uuv_y, uuv_z] == 0.0 and self.terrain_3d[uuv_x, uuv_y, uuv_z] > 0.5:
                            self.tl_table[enemy_y, uuv_x, uuv_y, uuv_z] = 120.0

        return self.tl_table

    def _move_robot(self, robot: Robot, action):
        """根据动作更新机器人位置，不考虑地形限制。地形避障也是学习的一部分。

        :param robot: 机器人对象
        :param action: 动作编号，0-5分别对应六个方向
        """

        dx, dy, dz = self.action_map.get(action, (0, 0, 0))
        robot.x = robot.x + dx
        robot.y = robot.y + dy
        robot.z = robot.z + dz

    def _enemy_step(self):
        """令敌方移动一步。目前采取的策略为，随机敌方初始化位置和朝向，然后按照巡逻的方式往返行动。
            
            为了加入一定的随机性，在每一步都有小概率原地不动.

            随机游动会使得 E(enemy_y) 方差过大，导致训练不稳定，因此暂时取消随机游动。可以考虑在轮数靠后的时候加入随机游动，增加环境的多样性。
        """
        if random.random() < 0.15:  # 有概率原地不动
            return

        new_y = self.enemy.y + self.enemy_forward_direction
        # 如果敌人到达巡逻边界，改变方向
        if new_y < self.enemy_y_min_idx or new_y > self.enemy_y_max_idx:
            self.enemy_forward_direction *= -1
            new_y = self.enemy.y + self.enemy_forward_direction * 2

        # 更新敌人位置
        if (self.terrain_3d[self.enemy.x, new_y, self.enemy.z] > 0.5):  # 确保新位置可通行
            self.enemy.y = new_y
        else:
            print_error(f"敌人尝试移动到不可通行位置: ({self.enemy.x}, {new_y}, {self.enemy.z}), 保持原地不动")

    def _query_TL(self, enemy_y, uuv_x, uuv_y, uuv_z):
        """查询传播损失 TL，根据我方机器人和敌方机器人的位置计算

        :param enemy_y: 敌方机器人 y 坐标
        :param uuv_x: 我方机器人 x 坐标
        :param uuv_y: 我方机器人 y 坐标
        :param uuv_z: 我方机器人 z 坐标
        :return: 传播损失 TL，单位dB
        """
        # 计算我方机器人和敌方机器人的距离
        return self.tl_table[enemy_y, uuv_x, uuv_y, uuv_z]

    def 计算隐蔽奖励(self, TL):
        """根据传播损失 TL 计算隐蔽奖励，TL 越大奖励越高

        :param TL: 传播损失，单位dB
        :return: 隐蔽奖励，数值越大表示越隐蔽
        """
        # 安全隐蔽区: TL > proper_TL + 3dB，奖励为正，且随着TL增加而增加
        temp = tanh((TL-66.0)/self.tl_tolerance) / self.recommended_steps
        if (temp > 0):
            return temp * 0.8
        else:
            return temp * 1.5

    def 计算靠近奖励(self, action):
        """根据动作计算靠近奖励，接近敌人奖励为正，远离敌人奖励为负

        :param action: 动作编号，0-5分别对应六个方向
        :return: 单步奖励，数值越大表示越接近敌人
        """

        # approach_reward = 0
        # approach_reward = -1 / self.recommended_steps * 0.5
        # if (action == 0):
        #     approach_reward = 1 / self.recommended_steps
        # else:
        #     approach_reward = -1 / self.recommended_steps

        # if (self.now_step > self.recommended_steps and approach_reward < 0):
        #     approach_reward *= 2

        # return approach_reward

        """基于势能差计算奖励：距离缩短给正分，距离拉大给负分"""
        # 假设目标是向更小的 X 坐标突进
        current_distance = abs(self.uuv.x - self.victory_x_idx)

        # 距离变化量：正数代表靠近了，负数代表远离了
        distance_diff = abs(self.last_uuv_location[0] - self.victory_x_idx) - current_distance

        # 给予奖励 (权重可以微调)
        approach_reward = distance_diff / self.recommended_steps * 3

        return approach_reward

    def 计算TL梯度奖励(self, last_TL, now_TL):
        """根据传播损失 TL 的变化计算梯度奖励，TL 下降奖励为正，TL 上升奖励为负

        :param last_TL: 上一步的传播损失，单位dB
        :param now_TL: 当前的传播损失，单位dB
        :return: TL梯度奖励，数值越大表示TL下降越多（更隐蔽）
        """
        return tanh(2 * (last_TL - now_TL)/self.tl_tolerance) / self.recommended_steps

    def 计算范围平均TL奖励(self):
        """计算以我方机器人为中心、边长为 field_of_view 的立方体范围内的平均 TL，并根据平均 TL 计算奖励

        :return: 范围平均 TL 奖励，数值越大表示范围内平均 TL 越高（更隐蔽）
        """
        half_fov = self.field_of_view // 2
        x_min = max(0, self.uuv.x - half_fov)
        x_max = min(self.map_width, self.uuv.x + half_fov + 1)
        y_min = max(0, self.uuv.y - half_fov)
        y_max = min(self.map_height, self.uuv.y + half_fov + 1)
        z_min = max(0, self.uuv.z - half_fov)
        z_max = min(self.map_depth, self.uuv.z + half_fov + 1)

        # 先计算中心小范围的 TL 均值，边长为 self.field_of_view // 4
        half_fov_small = self.field_of_view // 4
        x_min_small = max(0, self.uuv.x - half_fov_small)
        x_max_small = min(self.map_width, self.uuv.x + half_fov_small + 1)
        y_min_small = max(0, self.uuv.y - half_fov_small)
        y_max_small = min(self.map_height, self.uuv.y + half_fov_small + 1)
        z_min_small = max(0, self.uuv.z - half_fov_small)
        z_max_small = min(self.map_depth, self.uuv.z + half_fov_small + 1)

        core_tl_values = []
        temp = 0
        for z in range(z_min_small, z_max_small):
            for y in range(y_min_small, y_max_small):
                for x in range(x_min_small, x_max_small):
                    temp = self._query_TL(self.enemy.y, x, y, z)
                    if (temp > 0):
                        core_tl_values.append(temp)
        if len(core_tl_values) == 0:
            return 0

        temp = 0
        tl_values = []
        for z in range(z_min, z_max):
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    temp = self._query_TL(self.enemy.y, x, y, z)
                    if (temp > 0):
                        tl_values.append(temp)

        if len(tl_values) == 0:
            return 0

        average_tl = 0.7 * np.mean(core_tl_values) + 0.3 * np.mean(tl_values)
        self.区域平均TL = average_tl
        return tanh(1.5*(average_tl-66.0)/self.tl_tolerance) / self.recommended_steps

    def get_observation_tensor(self, device: str = "cuda"):
        """为 ACNet 生成观测张量（spatial_input 与 state_vector）。

        功能说明:
            生成 ACNet 所需的两路输入张量，所有计算直接在 GPU 执行：
            1. spatial_input: (1, 2, D, H, W) 的 float32 张量
               - 通道0: 传播损失 TL 热力图（围绕 UUV 的局部窗口）
               - 通道1: 地形可通行性（1.0=可通行, 0.0=不可通行，1=True, 0=False）
            2. state_vector: (1, N) 的 float32 张量（N 为配置中的 `state_vector_dim`）
            #    - (x_dist, y_uuv, z_uuv, y_enemy, current_tl, gradient_tl, average_tl) 归一化处理数值
               - (x_dist, y_uuv, z_uuv, y_enemy, current_tl, gradient_tl, average_tl, _cumulative_acoustic_signal) 归一化处理数值
               - 格外注意！所有数值全部是归一化处理过的！不直接代表物理数据！！

            坐标约定: 地形索引为 terrain_3d[x, y, z]，向量存储为 (x, y, z)。
            张量设备: 所有张量直接在指定设备（默认 cuda）上创建。

        输入参数:
            device (str): 计算设备，默认为 "cuda"。
            window_size (int): spatial 窗口半径（单位：网格单元数），默认为 16。
                最终 spatial_input 形状为 (1, 2, window_size, window_size, window_size)。

        输出参数:
            Tuple[torch.Tensor, torch.Tensor]:
                spatial_input: 形状 (1, 2, window_size, window_size, window_size)，dtype=float32，device=指定设备。
                state_vector: 形状 (1, N)，dtype=float32，device=指定设备（N 为 `state_vector_dim`）。

        调用示例:
            >>> spatial_input, state_vector = env.get_observation_tensor(device="cuda", window_size=16)
            >>> spatial_input.shape, state_vector.shape
            (torch.Size([1, 2, 16, 16, 16]), torch.Size([1, 8]))
        """

        import torch
        self.net_obervation_spatial_tensor.fill_(0)
        # 构建 UUV 观测向量
        # 第一维度：地形可通行性（1=不可通行，0=可通行）
        # 开始从地形数组中复制切片到观测数组中。复制大小：以UUV当前位置为中心，边长为 field_of_view 的立方体区域。Z轴厚度为 field_of_view_on_z。
        # 计算观测窗口在地图中的边界（以UUV位置为中心）
        half_fov_xy = self.field_of_view // 2
        half_fov_z = self.field_of_view_on_z // 2

        # 窗口在地图坐标系中的边界
        x_min_map = self.uuv.x - half_fov_xy
        x_max_map = self.uuv.x + half_fov_xy
        y_min_map = self.uuv.y - half_fov_xy
        y_max_map = self.uuv.y + half_fov_xy
        z_min_map = self.uuv.z - half_fov_z
        z_max_map = self.uuv.z + half_fov_z

        # 与地图实际范围的交集（防止越界）
        x_min_valid = max(0, x_min_map)
        x_max_valid = min(self.map_width - 1, x_max_map)
        y_min_valid = max(0, y_min_map)
        y_max_valid = min(self.map_height - 1, y_max_map)
        z_min_valid = max(0, z_min_map)
        z_max_valid = min(self.map_depth - 1, z_max_map)

        # 计算有效区域在观测张量中的偏移
        offset_x_min = x_min_valid - x_min_map
        offset_x_max = offset_x_min + (x_max_valid - x_min_valid + 1)
        offset_y_min = y_min_valid - y_min_map
        offset_y_max = offset_y_min + (y_max_valid - y_min_valid + 1)
        offset_z_min = z_min_valid - z_min_map
        offset_z_max = offset_z_min + (z_max_valid - z_min_valid + 1)

        # 从GPU张量中切片（注意地形索引顺序为 [x, y, z]，而TL索引为 [enemy_y, uuv_x, uuv_y, uuv_z]）
        # 通道0: TL热力图，需要从TL表中逐个查询
        # TL表索引: [enemy_y, uuv_x, uuv_y, uuv_z]
        tl_slice = self.tl_table_tensor[
            self.enemy.y,
            x_min_valid : x_max_valid + 1,
            y_min_valid : y_max_valid + 1,
            z_min_valid : z_max_valid + 1,
        ].float()

        self.net_obervation_spatial_tensor[0,
            offset_x_min : offset_x_max,
            offset_y_min : offset_y_max,
            offset_z_min : offset_z_max
        ] = tl_slice

        # 通道1: 地形可通行性（1=不可通行，0=可通行）
        terrain_slice = self.terrain_3d_tensor[
            x_min_valid : x_max_valid + 1,
            y_min_valid : y_max_valid + 1,
            z_min_valid : z_max_valid + 1,
        ].float()  # 转换为float32（True->1.0, False->0.0）

        # 将有效的地形数据填入预分配的观测张量（第二个通道）
        self.net_obervation_spatial_tensor[
            1,
            offset_x_min : offset_x_max,
            offset_y_min : offset_y_max,
            offset_z_min : offset_z_max,
        ] = terrain_slice

        # --- 构建 state_vector ---
        # --- 逻辑对齐：让“大数值”永远代表“好状态”
        # 梯度限幅与归一化，超过 5.0 的梯度统一视为“极大改善”，低于 -5.0 统一视为“极大恶化”
        clipped_grad = np.clip(self.当前TL - self.上次TL, -5.0, 5.0)
        clipped_grad = clipped_grad / 5.0

        # # X dist 零均值化
        # x_dist = (self.uuv.x - self.victory_x_idx) / (self.map_width - self.victory_x_idx)
        # x_dist = x_dist * 2 - 1
        # X dist 零均值化
        x_dist = 1 - (self.uuv.x - self.victory_x_idx) / (self.uuv_start_x_min_idx - self.victory_x_idx)

        # 累计观测部分，平方以增强靠近 1 时的差异化数值
        _cumulative_acoustic_signal = np.clip(self.cumulative_acoustic_signal, 0, self.tl_tolerance * 2.2) # 限幅，防止过大数值导致训练不稳定
        _cumulative_acoustic_signal = _cumulative_acoustic_signal / (self.tl_tolerance * 2.2)  # 归一化到 0-1 范围
        _cumulative_acoustic_signal = 1 - _cumulative_acoustic_signal  # 反转，使得接近被发现状态时数值接近0，远离被发现状态时数值接近1
        # 开方处理，使得接近 1 时的数值差异更小，远离 1 时的数值差异更大，增强模型对隐蔽状态的敏感度
        _cumulative_acoustic_signal = np.sqrt(np.clip(_cumulative_acoustic_signal, 0, 1))
        # 平方处理，使得接近 1 时的数值差异更大，远离 1 时的数值差异更小，增强模型对被发现状态的敏感度
        # _cumulative_acoustic_signal = _cumulative_acoustic_signal * _cumulative_acoustic_signal * np.sign(_cumulative_acoustic_signal)  # 保持符号

        # 累计步数部分，平方之以增强靠近推荐步数上限时的差异化数值
        step_ratio = 1.0 - (self.now_step / self.recommended_max_steps)
        # step_ratio = step_ratio * step_ratio

        # 状态向量
        state_vector_np = np.array(
            [
                [
                    float(x_dist),  # 与胜利位置的相对 x 坐标，并且进行归一化（0-1）
                    float(self.uuv.y / self.map_height),
                    float(self.uuv.z / self.map_depth),
                    float(self.enemy.y / self.map_height),
                    float(tanh((self.当前TL - 66.0) / (self.tl_tolerance * 1.5))),      # 当前 TL
                    float(clipped_grad),                                        # TL 梯度（经过限幅和归一化）
                    float(tanh((self.区域平均TL - 66.0) / (self.tl_tolerance * 1.5))),   # 范围平均 TL（经过归一化）
                    float(_cumulative_acoustic_signal),                          # 累计声学信号（经过归一化）
                    float(step_ratio)                                           # 步数比例（经过平方处理）
                ]
            ],
            dtype=np.float32,
        )
        self.net_state_vector_tensor = from_numpy(state_vector_np).to(device=device)

        # 对 self.net_obervation_spatial_tensor[0] 进行 tanh 归一化，增强数值稳定性
        self.net_obervation_spatial_tensor[0] = torch.tanh(
            (self.net_obervation_spatial_tensor[0] - 66.0) / (self.tl_tolerance * 1.5)
        )

        return self.net_obervation_spatial_tensor.unsqueeze(0), self.net_state_vector_tensor

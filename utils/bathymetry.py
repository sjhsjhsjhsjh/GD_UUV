"""
三维地形处理模块 - BTY文件解析、插值、重采样
支持Bellhop声学模型的地形数据读取和处理

功能：
  - 解析BTY格式地形文件
  - 使用三次样条插值建立连续地形函数
  - 按指定步长进行空间重采样
  - 支持垂直分层采样
  - 提供地形数据的查询接口
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from pathlib import Path
from typing import Tuple, Optional
from rich_print import print_info, print_warn, print_error


class BathymetryMap:
    """
    三维海底地形地图类 - 可通行性分析
    
    属性：
        nx_original: int
            原始地形网格在x方向的节点数
        ny_original: int
            原始地形网格在y方向的节点数
        x_range: tuple(float, float)
            原始地形x范围（单位：千米）
        y_range: tuple(float, float)
            原始地形y范围（单位：千米）
        depth_grid: np.ndarray
            原始深度数据矩阵，shape为(ny_original, nx_original)
        spline_func: scipy.interpolate.RectBivariateSpline or None
            三次样条插值函数（未调用interpolate()前为None）
        resampled_data: np.ndarray or None
            重采样后的三维可通行性布尔数组，shape为(ny_samples, nx_samples, nz_samples)
            True: 山体（不可通行），False: 水/空隙（可通行）
    
    示例：
        >>> bty_map = BathymetryMap()
        >>> bty_map.read_bty_file('qianshui.bty')
        >>> bty_map.interpolate_bathymetry()
        >>> bty_map.resample_terrain(x_step=0.1, y_step=0.1, z_step=0.05)
        >>> print(bty_map.resampled_data.shape)  # (101, 101, 11)
        >>> # 检查点(5km, 5km, 200m)是否为山体
        >>> is_mountain = bty_map.resampled_data[50, 50, 4]
        >>> can_pass = not is_mountain
    """
    
    def __init__(self):
        """初始化地形地图对象"""
        self.nx_original = None
        self.ny_original = None
        self.x_range = None
        self.y_range = None
        self.depth_grid = None
        self.spline_func = None
        self.resampled_data = None
        self.resample_params = {}
        
    def read_bty_file(self, file_path: str) -> bool:
        """
        读取并解析BTY格式文件
        
        BTY文件格式：
            R                    # 文件类型标识
            行数                  # 整数，表示y方向网格点数
            y_min y_max /        # y范围，单位为千米
            列数                  # 整数，表示x方向网格点数
            x_min x_max /        # x范围，单位为千米
            深度数据矩阵          # 浮点数矩阵，按行存储
        
        参数：
            file_path: str
                BTY文件的绝对路径或相对路径
        
        返回：
            bool: 读取成功返回True，失败返回False
        
        示例：
            >>> bty = BathymetryMap()
            >>> success = bty.read_bty_file('bellhop_example/qianshui.bty')
            >>> if success:
            ...     print(f"地形网格: {bty.ny_original}×{bty.nx_original}")
            ...     print(f"X范围: {bty.x_range[0]:.2f} - {bty.x_range[1]:.2f} km")
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            print_error(f"文件不存在: {file_path}")
            return False
            
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # 检查文件格式标识（移除引号和空白）
            first_line = lines[0].strip().strip("'\"")
            if first_line != 'R':
                print_error(f"BTY文件格式错误：文件头应为'R'，实际为'{first_line}'")
                return False
            
            # 读取行信息（y方向）
            self.ny_original = int(lines[1].strip())
            y_parts = lines[2].strip().split()
            self.y_range = (float(y_parts[0]), float(y_parts[1]))
            
            # 读取列信息（x方向）
            self.nx_original = int(lines[3].strip())
            x_parts = lines[4].strip().split()
            self.x_range = (float(x_parts[0]), float(x_parts[1]))
            
            # 读取深度数据
            data_lines = lines[5:]
            depth_values = []
            for line in data_lines:
                if line.strip():
                    values = [float(x) for x in line.split()]
                    depth_values.extend(values)
            
            # 验证数据完整性
            expected_points = self.nx_original * self.ny_original
            if len(depth_values) != expected_points:
                print_error(f"数据不完整：期望{expected_points}个点，实际{len(depth_values)}个")
                return False
            
            # 重新排列成矩阵（ny×nx，每行为x方向）
            self.depth_grid = np.array(depth_values).reshape(self.ny_original, self.nx_original)
            
            print_info(f"✓ 成功读取BTY文件: {file_path.name}")
            print_info(f"  网格大小: {self.ny_original}(Y) × {self.nx_original}(X)")
            print_info(f"  X范围: [{self.x_range[0]:.2f}, {self.x_range[1]:.2f}] km")
            print_info(f"  Y范围: [{self.y_range[0]:.2f}, {self.y_range[1]:.2f}] km")
            print_info(f"  深度范围: [{np.min(self.depth_grid):.2f}, {np.max(self.depth_grid):.2f}] m")
            
            return True
            
        except Exception as e:
            print_error(f"读取BTY文件失败: {str(e)}")
            return False
    
    def interpolate_bathymetry(self, kx: int = 3, ky: int = 3) -> bool:
        """
        使用三次样条插值建立地形函数
        
        使用scipy.interpolate.RectBivariateSpline进行二维三次样条插值。
        插值后可通过get_depth_at_point()快速查询任意位置的深度值。
        
        参数：
            kx: int，default=3
                x方向样条次数（1-5，默认3为三次样条）
            ky: int，default=3
                y方向样条次数（1-5，默认3为三次样条）
        
        返回：
            bool: 插值成功返回True，失败返回False
        
        示例：
            >>> bty = BathymetryMap()
            >>> bty.read_bty_file('qianshui.bty')
            >>> bty.interpolate_bathymetry()
            >>> # 现在可以查询任意点的深度
            >>> depth_at_center = bty.get_depth_at_point(5.0, 5.0)
        """
        if self.depth_grid is None:
            print_error("请先调用read_bty_file()读取地形数据")
            return False
        
        try:
            # 创建原始网格坐标
            x_coords = np.linspace(self.x_range[0], self.x_range[1], self.nx_original)
            y_coords = np.linspace(self.y_range[0], self.y_range[1], self.ny_original)
            
            # 构建三次样条插值函数
            # RectBivariateSpline(y, x, z)，注意y和x的顺序
            self.spline_func = RectBivariateSpline(
                y_coords, x_coords, self.depth_grid,
                kx=kx, ky=ky
            )
            
            print_info("✓ 三次样条插值构建完成")
            
            # 验证插值精度
            test_points = 5
            errors = []
            for i in np.linspace(0, self.ny_original - 1, test_points, dtype=int):
                for j in np.linspace(0, self.nx_original - 1, test_points, dtype=int):
                    x = x_coords[j]
                    y = y_coords[i]
                    original = self.depth_grid[i, j]
                    interpolated = float(self.spline_func(y, x)[0, 0])
                    errors.append(abs(original - interpolated))
            
            max_error = np.max(errors)
            mean_error = np.mean(errors)
            print_info(f"  插值验证（原始点）：最大误差={max_error:.4f}m，平均误差={mean_error:.4f}m")
            
            return True
            
        except Exception as e:
            print_error(f"插值失败: {str(e)}")
            return False
    
    def get_depth_at_point(self, x: float, y: float) -> Optional[float]:
        """
        查询任意位置的海深值
        
        通过已建立的样条插值函数查询点(x, y)处的深度值。
        注意：返回的是海深（正值），下游使用时可根据需要转为负值。
        
        参数：
            x: float
                查询点的x坐标（单位：千米），应在[x_min, x_max]范围内
            y: float
                查询点的y坐标（单位：千米），应在[y_min, y_max]范围内
        
        返回：
            float: 插值得到的深度值（单位：米，正值），范围应在原始深度范围内
            None: 如果查询失败
        
        示例：
            >>> bty = BathymetryMap()
            >>> bty.read_bty_file('qianshui.bty')
            >>> bty.interpolate_bathymetry()
            >>> depth = bty.get_depth_at_point(5.0, 5.0)
            >>> print(f"坐标(5km, 5km)的深度: {depth:.2f}m")
        """
        if self.spline_func is None:
            print_error("请先调用interpolate_bathymetry()建立插值函数")
            return None
        
        try:
            # RectBivariateSpline的调用格式为(y, x)
            depth = float(self.spline_func(y, x)[0, 0])
            return depth
        except Exception as e:
            print_error(f"查询点深度失败({x}, {y}): {str(e)}")
            return None
    
    def resample_terrain(self, x_step: float = 0.1, y_step: float = 0.1, 
                        z_step: float = 0.05) -> bool:
        """
        按指定步长对地形进行重采样，生成可通行性三维布尔数组
        
        在水平面上以(x_step, y_step)为间隔重采样，垂直方向以z_step为间隔分层。
        坐标系统：
            - 输入x_step, y_step, z_step单位为千米
            - 在x, y方向从原始范围的起点开始，按步长均匀采样
            - 在z方向（高度），从0开始每隔z_step米采一层
        
        输出数组形状为(ny_samples, nx_samples, nz_samples)的布尔数组，其中：
            - ny_samples = ceil((y_range[1] - y_range[0]) / y_step)
            - nx_samples = ceil((x_range[1] - x_range[0]) / x_step)
            - nz_samples = ceil(max_depth / z_step)
            - True: 该位置为山体（不可通行）
            - False: 该位置为水/空隙（可通行）
        
        参数：
            x_step: float，default=0.1
                x方向采样间隔，单位千米（通常0.1km=100m）
            y_step: float，default=0.1
                y方向采样间隔，单位千米（通常0.1km=100m）
            z_step: float，default=0.05
                z方向（高度）采样间隔，单位千米（通常0.05km=50m）
        
        返回：
            bool: 重采样成功返回True，失败返回False
        
        示例：
            >>> bty = BathymetryMap()
            >>> bty.read_bty_file('qianshui.bty')
            >>> bty.interpolate_bathymetry()
            >>> bty.resample_terrain(x_step=0.1, y_step=0.1, z_step=0.05)
            >>> print(bty.resampled_data.shape)  # (101, 101, 11)
            >>> print(f"山体点数：{np.count_nonzero(bty.resampled_data)}")
            >>> print(f"可通行点数：{np.count_nonzero(~bty.resampled_data)}")
        """
        if self.spline_func is None:
            print_error("请先调用interpolate_bathymetry()建立插值函数")
            return False
        
        try:
            self.resample_params = {
                'x_step': x_step,
                'y_step': y_step,
                'z_step': z_step
            }
            
            # 计算采样网格（使用linspace确保坐标在范围内）
            nx_samples = int(np.round((self.x_range[1] - self.x_range[0]) / x_step)) + 1
            ny_samples = int(np.round((self.y_range[1] - self.y_range[0]) / y_step)) + 1
            
            x_samples = np.linspace(self.x_range[0], self.x_range[1], nx_samples)
            y_samples = np.linspace(self.y_range[0], self.y_range[1], ny_samples)
            
            print_info(f"重采样参数：x_step={x_step}km, y_step={y_step}km, z_step={z_step}km")
            print_info(f"水平采样点数：{ny_samples}(Y) × {nx_samples}(X)")
            
            # 先在水平面采样所有点的深度
            xx, yy = np.meshgrid(x_samples, y_samples)
            
            # 使用ev()方法而不是直接调用，更安全
            depths_2d = self.spline_func.ev(yy, xx)
            
            # 保存2D海深网格为实例变量，用于后续保存到NPZ
            self.depths_2d = depths_2d
            
            # 计算垂直层数
            max_depth = np.max(depths_2d)
            nz_samples = int(np.ceil(max_depth / (z_step * 1000)))  # 转换为米
            
            print_info(f"最大深度：{max_depth:.2f}m，垂直分层数：{nz_samples}")
            
            # 初始化三维布尔数组（默认为False，表示可通行）
            self.resampled_data = np.zeros((ny_samples, nx_samples, nz_samples), dtype=bool)
            
            # 垂直采样：对于每个(x,y)点，标记其下方的山体
            z_step_m = z_step * 1000  # 转换为米
            z_layers = np.arange(z_step_m, (nz_samples + 1) * z_step_m, z_step_m)
            
            for i in range(ny_samples):
                for j in range(nx_samples):
                    depth = float(depths_2d[i, j])
                    # 对于这个位置，标记所有超过深度的z层为True（山体）
                    for k, z in enumerate(z_layers):
                        if z > depth:
                            self.resampled_data[i, j, k] = True  # 山体，不可通行
                        # 否则保持False（水，可通行）
            
            # 计算统计信息
            mountain_count = np.count_nonzero(self.resampled_data)
            passable_count = np.count_nonzero(~self.resampled_data)
            total_count = self.resampled_data.size
            
            print_info(f"✓ 重采样完成")
            print_info(f"  输出数组形状: {self.resampled_data.shape}")
            print_info(f"  山体点数: {mountain_count}/{total_count} ({100*mountain_count/total_count:.1f}%)")
            print_info(f"  可通行点数: {passable_count}/{total_count} ({100*passable_count/total_count:.1f}%)")
            
            return True
            
        except Exception as e:
            print_error(f"重采样失败: {str(e)}")
            return False
    
    def sample_vertical_layers(self, x: float, y: float, z_step: float = 0.05) -> Optional[np.ndarray]:
        """
        在指定位置进行垂直分层采样 - 获取该点的深度值
        
        在点(x, y)处从水表向下采样，按z_step间隔获取垂直方向的高度值。
        返回的高度值可用于确定可通行性的分层信息。
        
        参数：
            x: float
                查询位置的x坐标（单位：千米）
            y: float
                查询位置的y坐标（单位：千米）
            z_step: float，default=0.05
                采样间隔，单位千米（0.05km=50m）
        
        返回：
            np.ndarray: 形状为(n,)的一维数组，包含该位置的高度采样值（单位：米）
                       数组从z_step米开始，以z_step间隔递增
            None: 如果查询失败
        
        示例：
            >>> bty = BathymetryMap()
            >>> bty.read_bty_file('qianshui.bty')
            >>> bty.interpolate_bathymetry()
            >>> layers = bty.sample_vertical_layers(5.0, 5.0, z_step=0.05)
            >>> print(f"垂直采样点数: {len(layers)}")
            >>> print(f"深度范围: 0 - {layers[-1]:.2f}m")
            >>> # 对应的可通行性：在该位置，z_coords <= layers[-1]的地方都是可通行（False）
            >>> # 在该位置，z_coords > layers[-1]的地方都是不可通行（True）
        """
        if self.spline_func is None:
            print_error("请先调用interpolate_bathymetry()建立插值函数")
            return None
        
        try:
            # 查询该位置的深度
            depth = self.get_depth_at_point(x, y)
            if depth is None:
                return None
            
            # 生成垂直采样层
            z_step_m = z_step * 1000  # 转换为米
            z_layers = np.arange(z_step_m, depth + z_step_m/2, z_step_m)
            
            return z_layers
            
        except Exception as e:
            print_error(f"垂直采样失败({x}, {y}): {str(e)}")
            return None
    
    def save_resampled_data(self, output_path: str) -> bool:
        """
        将可通行性三维布尔数组和二维海深数据保存为npz格式
        
        npz文件包含以下数组和元数据：
            - 'terrain_3d': 三维可通行性布尔数组，shape=(ny_samples, nx_samples, nz_samples)
              * True: 该位置为山体（不可通行）
              * False: 该位置为水/空隙（可通行）
            - 'bathymetry_2d': 二维海深数组，shape=(ny_samples, nx_samples) [m]
              * 每个网格点的海底深度值（从插值获得）
            - 'x_coords': 水平面x坐标数组 [km]
            - 'y_coords': 水平面y坐标数组 [km]
            - 'z_coords': 高度坐标数组 [m]
            - 'x_range': 原始x范围 [km]
            - 'y_range': 原始y范围 [km]
            - 'max_depth': 最大深度 [m]
            - 'resample_params': 重采样参数字典
        
        参数：
            output_path: str
                输出文件路径（应以.npz结尾）
        
        返回：
            bool: 保存成功返回True，失败返回False
        
        示例：
            >>> bty = BathymetryMap()
            >>> bty.read_bty_file('qianshui.bty')
            >>> bty.interpolate_bathymetry()
            >>> bty.resample_terrain()
            >>> bty.save_resampled_data('output/terrain.npz')
            >>> 
            >>> # 加载并使用
            >>> data = np.load('output/terrain.npz')
            >>> terrain = data['terrain_3d']  # shape: (101, 101, 11), dtype: bool
            >>> can_pass = ~terrain[50, 50, 5]  # 中心位置第6层是否可通行
        """
        if self.resampled_data is None:
            print_error("请先调用resample_terrain()生成重采样数据")
            return False
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 准备坐标数组
            x_step = self.resample_params.get('x_step', 0.1)
            y_step = self.resample_params.get('y_step', 0.1)
            z_step = self.resample_params.get('z_step', 0.05)
            
            x_coords = np.linspace(self.x_range[0], self.x_range[1], self.resampled_data.shape[1])
            y_coords = np.linspace(self.y_range[0], self.y_range[1], self.resampled_data.shape[0])
            
            z_step_m = z_step * 1000
            z_coords = np.arange(z_step_m, (self.resampled_data.shape[2] + 1) * z_step_m, z_step_m)
            
            # 保存数据
            np.savez_compressed(
                output_path,
                terrain_3d=self.resampled_data,
                bathymetry_2d=self.depths_2d,  # 2D海深数组，shape=(ny_samples, nx_samples)，单位：米
                x_coords=x_coords,
                y_coords=y_coords,
                z_coords=z_coords,
                x_range=np.array(self.x_range),
                y_range=np.array(self.y_range),
                max_depth=z_coords[-1],  # 最大采样高度
                resample_params=self.resample_params
            )
            
            file_size = output_path.stat().st_size / (1024 * 1024)
            print_info(f"✓ 数据保存成功: {output_path.name}")
            print_info(f"  文件大小: {file_size:.2f} MB")
            print_info(f"  数据形状: {self.resampled_data.shape}")
            print_info(f"  数据类型: {self.resampled_data.dtype}")
            
            return True
            
        except Exception as e:
            print_error(f"保存数据失败: {str(e)}")
            return False


def parse_bty_file(file_path: str) -> Optional[BathymetryMap]:
    """
    便捷函数：一步到位读取和解析BTY文件
    
    这是BathymetryMap.read_bty_file()的包装函数，用于快速读取文件。
    返回的对象需要后续调用interpolate_bathymetry()和resample_terrain()处理。
    
    参数：
        file_path: str
            BTY文件路径
    
    返回：
        BathymetryMap: 初始化并读取完毕的地形对象
        None: 如果读取失败
    
    示例：
        >>> bty = parse_bty_file('bellhop_example/qianshui.bty')
        >>> if bty:
        ...     bty.interpolate_bathymetry()
        ...     bty.resample_terrain()
    """
    bty = BathymetryMap()
    if bty.read_bty_file(file_path):
        return bty
    else:
        return None

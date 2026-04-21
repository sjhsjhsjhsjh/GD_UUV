"""
地形可视化模块 - 三维地形显示和对比
支持原始地形、重采样地形的多种可视化方式
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Optional, Tuple
from utils.rich_print import print_info, print_warn, print_error


def plot_bathymetry_3d(bathy_map, output_path: Optional[str] = None, 
                       title: str = "Original Bathymetry 3D View", show: bool = True) -> bool:
    """
    绘制原始地形的三维表面图
    
    使用matplotlib的3D绘图功能显示海底地形的三维视图。
    每个网格点代表原始数据中的一个位置，深度值通过颜色和z轴高度表示。
    
    参数：
        bathy_map: BathymetryMap
            地形地图对象，应已调用read_bty_file()方法
        output_path: str, optional
            如果提供，将图像保存到此路径（如 'output/terrain_3d.png'）
            如果为None且show=True，仅显示图像但不保存
        title: str
            图表标题，默认为"原始海底地形三维图"
        show: bool, default=True
            是否显示GUI窗口，默认True（显示）
    
    返回：
        bool: 绘制成功返回True，失败返回False
    
    示例：
        >>> from utils.bathymetry import parse_bty_file
        >>> baty = parse_bty_file('qianshui.bty')
        >>> baty.interpolate_bathymetry()
        >>> plot_bathymetry_3d(baty, show=True)  # 显示窗口
        >>> plot_bathymetry_3d(baty, output_path='output/original_3d.png', show=False)  # 仅保存
    """
    if bathy_map.depth_grid is None:
        print_error("地形对象未读取数据，请先调用read_bty_file()")
        return False
    
    try:
        # 创建网格坐标
        x_coords = np.linspace(bathy_map.x_range[0], bathy_map.x_range[1], 
                               bathy_map.nx_original)
        y_coords = np.linspace(bathy_map.y_range[0], bathy_map.y_range[1], 
                               bathy_map.ny_original)
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        # 创建图表
        fig = plt.figure(figsize=(12, 8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制表面
        surf = ax.plot_surface(xx, yy, bathy_map.depth_grid, 
                              cmap='ocean', alpha=0.8, 
                              linewidth=0, antialiased=True)
        
        # 设置标签和标题
        ax.set_xlabel('X (km)', fontsize=10)
        ax.set_ylabel('Y (km)', fontsize=10)
        ax.set_zlabel('Depth (m)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # 添加颜色条
        fig.colorbar(surf, ax=ax, label='Depth (m)', shrink=0.5)
        
        # 调整视角
        ax.view_init(elev=25, azim=45)
        
        # 显示或保存
        if show:
            plt.show()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            print_info(f"✓ 3D图像已保存: {output_path.name}")
        
        plt.close()
        return True
        
    except Exception as e:
        print_error(f"3D绘图失败: {str(e)}")
        return False


def plot_bathymetry_2d(bathy_map, output_path: Optional[str] = None,
                       title: str = "Bathymetry Contour Map", levels: int = 20, show: bool = True) -> bool:
    """
    绘制地形的二维等高线图
    
    以俯视角显示海底地形，使用等高线表示深度分布。
    适合观察地形的整体分布和梯度变化。
    
    参数：
        bathy_map: BathymetryMap
            地形地图对象，应已调用read_bty_file()方法
        output_path: str, optional
            输出路径，如'output/terrain_2d.png'，为None时取决于show参数
        title: str
            图表标题
        levels: int
            等高线的数量，default=20
        show: bool, default=True
            是否显示GUI窗口
    
    返回：
        bool: 成功返回True
    
    示例：
        >>> baty = parse_bty_file('qianshui.bty')
        >>> baty.interpolate_bathymetry()
        >>> plot_bathymetry_2d(baty, show=True)
    """
    if bathy_map.depth_grid is None:
        print_error("地形对象未读取数据")
        return False
    
    try:
        # 创建网格坐标
        x_coords = np.linspace(bathy_map.x_range[0], bathy_map.x_range[1], 
                               bathy_map.nx_original)
        y_coords = np.linspace(bathy_map.y_range[0], bathy_map.y_range[1], 
                               bathy_map.ny_original)
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        
        # 绘制等高线图
        contourf = ax.contourf(xx, yy, bathy_map.depth_grid, 
                               levels=levels, cmap='ocean')
        contour = ax.contour(xx, yy, bathy_map.depth_grid, 
                            levels=levels, colors='black', alpha=0.3, linewidths=0.5)
        
        # 添加等高线标签
        ax.clabel(contour, inline=True, fontsize=8, fmt='%1.0f')
        
        # 设置标签
        ax.set_xlabel('X (km)', fontsize=10)
        ax.set_ylabel('Y (km)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        
        # 添加颜色条
        cbar = fig.colorbar(contourf, ax=ax, label='Depth (m)')
        
        # 显示或保存
        if show:
            plt.show()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            print_info(f"✓ 2D等高线图已保存: {output_path.name}")
        
        plt.close()
        return True
        
    except Exception as e:
        print_error(f"2D绘图失败: {str(e)}")
        return False


def plot_vertical_profile(bathy_map, x: float, y: float,
                         output_path: Optional[str] = None, show: bool = True) -> bool:
    """
    绘制指定位置的垂直深度轮廓
    
    显示从水表到海底的深度变化（虽然通常是垂直线条，但可用于验证插值）
    
    参数：
        bathy_map: BathymetryMap
            地形对象，应已调用interpolate_bathymetry()
        x: float
            查询位置x坐标 (km)
        y: float
            查询位置y坐标 (km)
        output_path: str, optional
            输出路径
        show: bool, default=True
            是否显示GUI窗口
    
    返回：
        bool: 成功返回True
    
    示例：
        >>> baty = parse_bty_file('qianshui.bty')
        >>> baty.interpolate_bathymetry()
        >>> plot_vertical_profile(baty, x=5.0, y=5.0, show=True)
    """
    if bathy_map.spline_func is None:
        print_error("请先调用interpolate_bathymetry()")
        return False
    
    try:
        # 查询该位置的深度和周围点
        depth = bathy_map.get_depth_at_point(x, y)
        
        # 生成周围点的深度来显示梯度
        radius = 0.5  # km
        x_range = np.linspace(x - radius, x + radius, 50)
        y_range = np.linspace(y - radius, y + radius, 50)
        
        depths_x = [bathy_map.get_depth_at_point(xi, y) for xi in x_range]
        depths_y = [bathy_map.get_depth_at_point(x, yi) for yi in y_range]
        
        # 创建子图
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=100)
        
        # X-direction profile
        axes[0].plot(x_range, depths_x, 'b-', linewidth=2, label='Depth')
        axes[0].scatter([x], [depth], color='r', s=100, zorder=5, label='Query Point')
        axes[0].set_xlabel('X (km)', fontsize=10)
        axes[0].set_ylabel('Depth (m)', fontsize=10)
        axes[0].set_title(f'X-Direction Depth Profile (Y={y:.2f}km)', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Y-direction profile
        axes[1].plot(y_range, depths_y, 'g-', linewidth=2, label='Depth')
        axes[1].scatter([y], [depth], color='r', s=100, zorder=5, label='Query Point')
        axes[1].set_xlabel('Y (km)', fontsize=10)
        axes[1].set_ylabel('Depth (m)', fontsize=10)
        axes[1].set_title(f'Y-Direction Depth Profile (X={x:.2f}km)', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            print_info(f"✓ 深度剖面图已保存: {output_path.name}")
        
        plt.close()
        return True
        
    except Exception as e:
        print_error(f"剖面绘图失败: {str(e)}")
        return False


def compare_original_resampled(bathy_map, output_dir: Optional[str] = None, show: bool = True) -> bool:
    """
    对比显示原始地形和重采样后地形
    
    并排显示原始地形（200×200）和重采样后地形在水平面上的深度分布。
    用于验证重采样的效果和插值的准确性。
    
    参数：
        bathy_map: BathymetryMap
            地形对象，应已调用resample_terrain()
        output_dir: str, optional
            输出目录，为None时取决于show参数，默认'output'
        show: bool, default=True
            是否显示GUI窗口
    
    返回：
        bool: 成功返回True
    
    示例：
        >>> baty = parse_bty_file('qianshui.bty')
        >>> baty.interpolate_bathymetry()
        >>> baty.resample_terrain()
        >>> compare_original_resampled(baty, show=True)
    """
    if bathy_map.resampled_data is None:
        print_error("请先调用resample_terrain()生成重采样数据")
        return False
    
    try:
        # 准备原始数据
        x_orig = np.linspace(bathy_map.x_range[0], bathy_map.x_range[1], 
                             bathy_map.nx_original)
        y_orig = np.linspace(bathy_map.y_range[0], bathy_map.y_range[1], 
                             bathy_map.ny_original)
        xx_orig, yy_orig = np.meshgrid(x_orig, y_orig)
        
        # 准备重采样数据（取第一层）
        resampled_2d = bathy_map.resampled_data[:, :, 0]
        x_resamp = np.linspace(bathy_map.x_range[0], bathy_map.x_range[1], 
                               bathy_map.resampled_data.shape[1])
        y_resamp = np.linspace(bathy_map.y_range[0], bathy_map.y_range[1], 
                               bathy_map.resampled_data.shape[0])
        xx_resamp, yy_resamp = np.meshgrid(x_resamp, y_resamp)
        
        # 创建对比图
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=100)
        
        # Original terrain
        im1 = axes[0].contourf(xx_orig, yy_orig, bathy_map.depth_grid, 
                              levels=20, cmap='ocean')
        axes[0].set_xlabel('X (km)', fontsize=10)
        axes[0].set_ylabel('Y (km)', fontsize=10)
        axes[0].set_title(f'Original Terrain ({bathy_map.ny_original}×{bathy_map.nx_original})', 
                         fontsize=11, fontweight='bold')
        axes[0].set_aspect('equal')
        fig.colorbar(im1, ax=axes[0], label='Depth (m)')
        
        # Resampled terrain
        im2 = axes[1].contourf(xx_resamp, yy_resamp, resampled_2d, 
                              levels=20, cmap='ocean')
        axes[1].set_xlabel('X (km)', fontsize=10)
        axes[1].set_ylabel('Y (km)', fontsize=10)
        axes[1].set_title(f'Resampled Terrain ({resampled_2d.shape[0]}×{resampled_2d.shape[1]})', 
                         fontsize=11, fontweight='bold')
        axes[1].set_aspect('equal')
        fig.colorbar(im2, ax=axes[1], label='Depth (m)')
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / 'comparison_original_resampled.png'
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            print_info(f"✓ 对比图已保存: {output_path.name}")
        
        plt.close()
        return True
        
    except Exception as e:
        print_error(f"对比绘图失败: {str(e)}")
        return False


def plot_resampled_3d_sample(bathy_map, output_path: Optional[str] = None, show: bool = True) -> bool:
    """
    绘制重采样数据的3D样本（第一层）
    
    将重采样后的数据（通常是二维平面）显示为3D图形以验证数据质量。
    仅绘制第一层（最浅的采样层），用于快速验证。
    
    参数：
        bathy_map: BathymetryMap
            地形对象，应已调用resample_terrain()
        output_path: str, optional
            输出路径
        show: bool, default=True
            是否显示GUI窗口
    
    返回：
        bool: 成功返回True
    
    示例：
        >>> baty = parse_bty_file('qianshui.bty')
        >>> baty.interpolate_bathymetry()
        >>> baty.resample_terrain()
        >>> plot_resampled_3d_sample(baty, show=True)
    """
    if bathy_map.resampled_data is None:
        print_error("请先调用resample_terrain()")
        return False
    
    try:
        # 取第一层数据
        data_2d = bathy_map.resampled_data[:, :, 0]
        
        # 创建网格
        x_step = bathy_map.resample_params.get('x_step', 0.1)
        y_step = bathy_map.resample_params.get('y_step', 0.1)
        
        x_coords = np.arange(bathy_map.x_range[0], bathy_map.x_range[1] + x_step/2, x_step)
        y_coords = np.arange(bathy_map.y_range[0], bathy_map.y_range[1] + y_step/2, y_step)
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        # 创建3D图
        fig = plt.figure(figsize=(12, 8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制表面（使用有效数据，NaN自动排除）
        surf = ax.plot_surface(xx, yy, data_2d, cmap='ocean', 
                              alpha=0.8, linewidth=0, antialiased=True)
        
        # Labels and title
        ax.set_xlabel('X (km)', fontsize=10)
        ax.set_ylabel('Y (km)', fontsize=10)
        ax.set_zlabel('Depth (m)', fontsize=10)
        ax.set_title('Resampled Terrain 3D View (Layer 1)', fontsize=12, fontweight='bold')
        
        # Colorbar
        fig.colorbar(surf, ax=ax, label='Depth (m)', shrink=0.5)
        
        # 视角
        ax.view_init(elev=25, azim=45)
        
        if show:
            plt.show()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            print_info(f"✓ 重采样3D图已保存: {output_path.name}")
        
        plt.close()
        return True
        
    except Exception as e:
        print_error(f"3D重采样图绘制失败: {str(e)}")
        return False

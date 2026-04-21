#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BTY文件处理命令行工具
功能：读取BTY文件 -> 插值 -> 重采样 -> 保存为npz

用法：
    # 使用配置文件运行（推荐）
    python scripts/process_bty.py
    
    # 或指定自定义参数
    python scripts/process_bty.py --input bellhop_example/qianshui.bty --output output/terrain.npz
    python scripts/process_bty.py --input qianshui.bty --output output/ --x-step 100 --y-step 100 --z-step 50
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.bathymetry import parse_bty_file
from utils.rich_print import print_info, print_warn, print_error

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def process_bty_file(input_path: str, output_path: str,
                     x_step: float = 100.0, y_step: float = 100.0, 
                     z_step: float = 50.0, show_visualization: bool = False) -> bool:
    """
    完整的BTY文件处理流程
    
    从读取原始文件到保存重采样数据的完整工作流。
    
    参数：
        input_path: str
            输入BTY文件路径
        output_path: str
            输出npz文件路径
        x_step: float
            x方向采样步长（单位：米），default=100
        y_step: float
            y方向采样步长（单位：米），default=100
        z_step: float
            z方向采样步长（单位：米），default=50
        show_visualization: bool
            是否显示可视化GUI窗口，default=False
    
    返回：
        bool: 处理成功返回True
    
    示例：
        >>> success = process_bty_file(
        ...     input_path='bellhop_example/qianshui.bty',
        ...     output_path='output/terrain.npz',
        ...     x_step=100, y_step=100, z_step=50,
        ...     show_visualization=True
        ... )
    """
    
    # 转换单位：米 -> 千米
    x_step_km = x_step / 1000.0
    y_step_km = y_step / 1000.0
    z_step_km = z_step / 1000.0
    
    print_info(f"开始处理BTY文件: {input_path}")
    print_info(f"采样参数: x={x_step}m, y={y_step}m, z={z_step}m")
    
    # 步骤1：读取文件
    print_info("\n[1/4] 读取BTY文件...")
    bathy_map = parse_bty_file(input_path)
    if not bathy_map:
        print_error("读取BTY文件失败")
        return False
    
    # 步骤2：建立插值函数
    print_info("\n[2/4] 建立三次样条插值...")
    if not bathy_map.interpolate_bathymetry():
        print_error("插值失败")
        return False
    
    # 步骤3：重采样
    print_info("\n[3/4] 重采样地形数据...")
    if not bathy_map.resample_terrain(x_step=x_step_km, y_step=y_step_km, z_step=z_step_km):
        print_error("重采样失败")
        return False
    
    # 步骤4：保存数据
    print_info("\n[4/4] 保存重采样数据...")
    if not bathy_map.save_resampled_data(output_path):
        print_error("保存数据失败")
        return False
    
    # 可选：显示可视化
    if show_visualization:
        print_info("\n[5/4] 生成可视化...")
        try:
            from utils.terrain_visualization import (
                plot_bathymetry_3d, plot_bathymetry_2d,
                compare_original_resampled, plot_resampled_3d_sample
            )
            
            viz_list = [
                ("原始地形3D", lambda: plot_bathymetry_3d(bathy_map, show=True)),
                ("地形2D等高线", lambda: plot_bathymetry_2d(bathy_map, show=True)),
                ("原始vs重采样对比", lambda: compare_original_resampled(bathy_map, show=True)),
                ("重采样3D样本", lambda: plot_resampled_3d_sample(bathy_map, show=True)),
            ]
            
            for name, func in viz_list:
                try:
                    func()
                except Exception as e:
                    print_warn(f"  • {name} 失败: {str(e)}")
        except ImportError:
            print_warn("  • 无法加载可视化模块，跳过")
    
    print_info("\n" + "="*60)
    print_info("✓ 处理完成！")
    print_info("="*60)
    
    return True


def load_config_params():
    """
    从配置文件中加载BTY处理参数
    
    返回：
        dict: 包含参数的字典，keys为(input_path, output_path, x_step, y_step, z_step)
        None: 如果配置文件不存在或加载失败
    """
    if not HAS_YAML:
        return None
    
    config_path = project_root / 'configs' / 'main_config.yaml'
    
    if not config_path.exists():
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if 'env' not in config:
            return None
        
        env_config = config['env']
        
        # 构建完整的BTY文件路径
        bty_filename = env_config.get('bty_file_name', 'qianshui.bty')
        input_path = project_root / 'bellhop_example' / bty_filename
        
        # 获取输出路径
        output_path = project_root / env_config.get('terrain_output_path', 'output/terrain.npz')
        
        # 获取采样参数（已经是米）
        x_step = env_config.get('sampling_x_step', 100)
        y_step = env_config.get('sampling_y_step', 100)
        z_step = env_config.get('sampling_z_step', 50)
        
        return {
            'input_path': str(input_path),
            'output_path': str(output_path),
            'x_step': float(x_step),
            'y_step': float(y_step),
            'z_step': float(z_step)
        }
    except Exception as e:
        print_warn(f"读取配置文件失败: {str(e)}")
        return None


def main():
    """
    命令行入口函数
    
    解析命令行参数并调用process_bty_file()进行处理
    """
    parser = argparse.ArgumentParser(
        description='BTY文件处理工具：读取、插值、重采样、保存',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 使用配置文件参数运行（推荐）
  python scripts/process_bty.py
  
  # 使用配置文件参数并显示可视化
  python scripts/process_bty.py --show
  
  # 使用默认采样参数（100×100×50 米）
  python scripts/process_bty.py --input bellhop_example/qianshui.bty --output output/terrain.npz
  
  # 指定自定义采样参数并显示可视化
  python scripts/process_bty.py \\
      --input qianshui.bty \\
      --output output/custom_terrain.npz \\
      --x-step 200 --y-step 200 --z-step 100 \\
      --show
  
  # 输出到目录（自动生成文件名）
  python scripts/process_bty.py --input qianshui.bty --output output/
        """
    )
    
    parser.add_argument('-i', '--input', type=str, default=None,
                       help='输入BTY文件路径（可选，默认从配置文件读取）')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='输出npz文件路径或目录（可选，默认从配置文件读取）')
    parser.add_argument('--x-step', type=float, default=None,
                       help='X方向采样步长，单位米（可选，默认从配置文件读取）')
    parser.add_argument('--y-step', type=float, default=None,
                       help='Y方向采样步长，单位米（可选，默认从配置文件读取）')
    parser.add_argument('--z-step', type=float, default=None,
                       help='Z方向采样步长，单位米（可选，默认从配置文件读取）')
    parser.add_argument('--show', action='store_true', default=False,
                       help='处理完成后显示可视化GUI窗口')
    
    args = parser.parse_args()
    
    # 尝试从配置文件读取参数
    config_params = load_config_params()
    
    # 命令行参数优先级高于配置文件
    input_path = args.input
    output_path = args.output
    x_step = args.x_step
    y_step = args.y_step
    z_step = args.z_step
    
    # 如果未指定命令行参数，使用配置文件的值
    if config_params:
        if input_path is None:
            input_path = config_params['input_path']
        if output_path is None:
            output_path = config_params['output_path']
        if x_step is None:
            x_step = config_params['x_step']
        if y_step is None:
            y_step = config_params['y_step']
        if z_step is None:
            z_step = config_params['z_step']
    
    # 如果还是没有参数，使用默认值
    if input_path is None:
        input_path = str(project_root / 'bellhop_example' / 'qianshui.bty')
    if output_path is None:
        output_path = str(project_root / 'output' / 'bty' / 'terrain.npz')
    if x_step is None:
        x_step = 100.0
    if y_step is None:
        y_step = 100.0
    if z_step is None:
        z_step = 50.0
    
    # 处理输出路径（如果是目录则自动生成文件名）
    if output_path.endswith('/') or output_path.endswith('\\'):
        output_path = Path(output_path) / 'terrain_resampled.npz'
    elif Path(output_path).is_dir() or (not output_path.endswith(('.npz', '/'))):
        if not output_path.endswith(('.npz', '/')):
            output_path = Path(output_path) / 'terrain_resampled.npz'
    
    # 执行处理
    success = process_bty_file(
        input_path=input_path,
        output_path=str(output_path),
        x_step=x_step,
        y_step=y_step,
        z_step=z_step,
        show_visualization=args.show
    )
    
    # 返回退出码
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

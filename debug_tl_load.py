#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
调试 TL 数据加载和过滤
"""

import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.rich_print import print_info, print_warn, print_error

def debug_tl_load():
    """
    调试 TL 数据加载
    """
    # 从 TL 数据文件读取
    tl_file = PROJECT_ROOT / 'TLdata' / 'average_TL_results.txt'
    
    if not tl_file.exists():
        print_error(f"TL文件不存在: {tl_file}")
        return
    
    print_info(f"读取 TL 文件: {tl_file}")
    
    # 读取数据
    rows = []
    with open(tl_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                values = [float(x) for x in line.replace(',', ' ').split()]
                if len(values) >= 5:
                    rows.append(values[:5])
                    if i < 5:
                        print_info(f"样本行 {i}: {values[:5]}")
            except ValueError as e:
                if i < 5:
                    print_warn(f"行 {i} 解析错误: {line[:50]}")
                continue
    
    if not rows:
        print_error("没有有效数据行")
        return
    
    rows = np.array(rows, dtype=np.float32)
    print_info(f"总行数: {len(rows)}")
    
    # 检查采样参数
    sampling_x_step = 100
    sampling_y_step = 100
    sampling_z_step = 50
    
    print_info(f"采样参数: x={sampling_x_step}, y={sampling_y_step}, z={sampling_z_step}")
    
    # 转换网格坐标
    print_info("\n=== 分析列数据 ===")
    for col_idx in range(min(5, rows.shape[1])):
        col_data = rows[:, col_idx]
        print_info(f"列 {col_idx}: min={col_data.min():.1f}, max={col_data.max():.1f}, "
                   f"mean={col_data.mean():.1f}, unique_count={len(np.unique(col_data))}")
    
    # 尝试不同的列定义
    print_info("\n=== 尝试识别列含义 ===")
    
    # 假设: enemy_y_grid, uuv_x_grid, uuv_y_grid, uuv_z_grid, tl_db
    try:
        enemy_y_km_vals = rows[:, 0] * sampling_y_step / 1000.0
        uuv_x_km_vals = rows[:, 1] * sampling_x_step / 1000.0
        uuv_y_km_vals = rows[:, 2] * sampling_y_step / 1000.0
        uuv_z_m_vals = rows[:, 3] * sampling_z_step
        tl_vals = rows[:, 4]
        
        print_info(f"敌方Y (km): min={enemy_y_km_vals.min():.3f}, max={enemy_y_km_vals.max():.3f}")
        print_info(f"UUV X (km): min={uuv_x_km_vals.min():.3f}, max={uuv_x_km_vals.max():.3f}")
        print_info(f"UUV Y (km): min={uuv_y_km_vals.min():.3f}, max={uuv_y_km_vals.max():.3f}")
        print_info(f"UUV Z (m): min={uuv_z_m_vals.min():.1f}, max={uuv_z_m_vals.max():.1f}")
        print_info(f"TL (dB): min={tl_vals.min():.1f}, max={tl_vals.max():.1f}")
        
        # 测试过滤
        test_enemy_y = 4.0
        print_info(f"\n=== 测试过滤: enemy_y={test_enemy_y} km ===")
        mask = np.abs(enemy_y_km_vals - test_enemy_y) <= 0.5
        print_info(f"容差±0.5km内的点数: {np.sum(mask)}")
        
        # 显示敌方Y值的分布
        unique_enemy_y = np.unique(enemy_y_km_vals)
        print_info(f"\n敌方Y唯一值 ({len(unique_enemy_y)}个): {unique_enemy_y[:10]}...")
        
        # 查看 enemy_y=4.0 附近的值
        close_to_4 = enemy_y_km_vals[(enemy_y_km_vals > 3.9) & (enemy_y_km_vals < 4.1)]
        print_info(f"3.9-4.1范围内的敌方Y值: {len(close_to_4)} 个, 值={np.unique(close_to_4)}")
        
    except Exception as e:
        print_error(f"分析错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    debug_tl_load()

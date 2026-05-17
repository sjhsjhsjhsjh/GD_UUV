#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试地形和TL数据加载
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from omegaconf import OmegaConf

def load_terrain_data():
    """加载地形数据"""
    try:
        terrain_npz_path = PROJECT_ROOT / 'output' / 'bty' / 'terrain.npz'
        print(f"🔍 查找地形文件: {terrain_npz_path}")
        if not terrain_npz_path.exists():
            print(f"❌ 地形文件不存在")
            return None
        
        data = np.load(terrain_npz_path)
        bathymetry_2d = data['bathymetry_2d'].astype(np.float32)
        x_coords = data['x_coords'].astype(np.float32)
        y_coords = data['y_coords'].astype(np.float32)
        
        print(f"✅ 地形数据加载成功!")
        print(f"   - Shape: {bathymetry_2d.shape}")
        print(f"   - X range: {x_coords[0]:.2f} ~ {x_coords[-1]:.2f}")
        print(f"   - Y range: {y_coords[0]:.2f} ~ {y_coords[-1]:.2f}")
        
        return {
            'x_km': x_coords.tolist(),
            'y_km': y_coords.tolist(),
            'depth_m': bathymetry_2d.tolist(),
            'shape': list(bathymetry_2d.shape)
        }
    
    except Exception as e:
        print(f"❌ 加载地形数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_tl_data(enemy_y_km=4.0):
    """加载TL数据"""
    try:
        # 加载配置
        main_cfg_path = PROJECT_ROOT / 'configs' / 'main_config.yaml'
        main_cfg = OmegaConf.load(main_cfg_path)
        
        tl_file = PROJECT_ROOT / 'TLdata' / 'average_TL_results.txt'
        print(f"🔍 查找TL文件: {tl_file}")
        
        if not tl_file.exists():
            print(f"❌ TL文件不存在")
            return None
        
        # 读取TL数据
        rows = []
        for line in tl_file.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                values = [float(x) for x in line.replace(',', ' ').split()]
                if len(values) >= 5:
                    rows.append(values[:5])
            except ValueError:
                continue
        
        if not rows:
            print(f"❌ TL文件没有有效数据")
            return None
        
        rows = np.array(rows, dtype=np.float32)
        
        # 获取采样步长
        sampling_x_step = float(main_cfg.env.sampling_x_step)
        sampling_y_step = float(main_cfg.env.sampling_y_step)
        sampling_z_step = float(main_cfg.env.sampling_z_step)
        tl_color_cap_db = 120
        
        print(f"✅ TL文件加载成功!")
        print(f"   - 总行数: {len(rows)}")
        print(f"   - 采样步长: x={sampling_x_step}, y={sampling_y_step}, z={sampling_z_step}")
        
        # 转换网格坐标为实际距离
        enemy_y_km_vals = rows[:, 0] * sampling_y_step / 1000.0
        uuv_x_km_vals = rows[:, 1] * sampling_x_step / 1000.0
        uuv_y_km_vals = rows[:, 2] * sampling_y_step / 1000.0
        uuv_z_m_vals = rows[:, 3] * sampling_z_step
        tl_vals = rows[:, 4]
        
        # 将0 dB替换为限幅值
        tl_vals = np.where(tl_vals == 0, tl_color_cap_db, tl_vals)
        
        # 筛选与当前敌方Y坐标相关的数据
        mask = np.abs(enemy_y_km_vals - enemy_y_km) <= 0.5
        
        print(f"   - 敌方Y: {enemy_y_km} km，筛选范围: ±0.5 km")
        print(f"   - 符合条件的点数: {np.sum(mask)}")
        
        return {
            'x_km': uuv_x_km_vals[mask].tolist(),
            'y_km': uuv_y_km_vals[mask].tolist(),
            'z_m': uuv_z_m_vals[mask].tolist(),
            'tl_db': tl_vals[mask].tolist(),
            'stats': {
                'valid_count': int(np.sum(mask)),
                'tl_min_db': float(np.min(tl_vals[mask])) if np.sum(mask) > 0 else 0,
                'tl_max_db': float(np.max(tl_vals[mask])) if np.sum(mask) > 0 else 0,
                'tl_mean_db': float(np.mean(tl_vals[mask])) if np.sum(mask) > 0 else 0,
            }
        }
    
    except Exception as e:
        print(f"❌ 加载TL数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    print("=" * 60)
    print("地形和TL数据加载测试")
    print("=" * 60)
    
    print("\n1️⃣ 加载地形数据...")
    terrain = load_terrain_data()
    
    print("\n2️⃣ 加载TL数据 (敌方Y=4.0 km)...")
    tl = load_tl_data(4.0)
    
    print("\n" + "=" * 60)
    if terrain and tl:
        print("✅ 所有数据加载成功!")
    else:
        print("❌ 数据加载失败")
    print("=" * 60)

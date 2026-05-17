#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 server.py 中的函数
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 复制 server.py 中的函数
import numpy as np
from omegaconf import OmegaConf

def _load_terrain_data(env) -> dict:
    """
    从环境对象提取地形数据
    """
    try:
        # 加载地形 NPZ 文件
        terrain_npz_path = PROJECT_ROOT / 'output' / 'bty' / 'terrain.npz'
        print(f"加载地形: {terrain_npz_path}, 存在: {terrain_npz_path.exists()}")
        if not terrain_npz_path.exists():
            print(f"地形文件不存在: {terrain_npz_path}")
            return None
        
        data = np.load(terrain_npz_path)
        bathymetry_2d = data['bathymetry_2d'].astype(np.float32)
        x_coords = data['x_coords'].astype(np.float32)
        y_coords = data['y_coords'].astype(np.float32)
        
        result = {
            'x_km': x_coords.tolist(),
            'y_km': y_coords.tolist(),
            'depth_m': bathymetry_2d.tolist(),
            'shape': list(bathymetry_2d.shape)
        }
        print(f"✅ 地形数据加载成功, shape: {bathymetry_2d.shape}")
        return result
    
    except Exception as e:
        print(f"❌ 加载地形数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def _load_tl_data(cfg, enemy_y_km: float) -> dict:
    """
    从配置中加载TL数据
    """
    try:
        # 从配置中获取TL文件路径
        tl_file = PROJECT_ROOT / 'TLdata' / 'average_TL_results.txt'
        print(f"加载TL: {tl_file}, 存在: {tl_file.exists()}")
        
        if not tl_file.exists():
            print(f"TL文件不存在: {tl_file}")
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
        
        print(f"读取 {len(rows)} 行 TL 数据")
        if not rows:
            print(f"TL文件没有有效数据: {tl_file}")
            return None
        
        rows = np.array(rows, dtype=np.float32)
        
        # 获取采样步长
        try:
            sampling_x_step = float(cfg.env.sampling_x_step) if hasattr(cfg, 'env') else 100
            sampling_y_step = float(cfg.env.sampling_y_step) if hasattr(cfg, 'env') else 100
            sampling_z_step = float(cfg.env.sampling_z_step) if hasattr(cfg, 'env') else 50
        except:
            sampling_x_step = 100
            sampling_y_step = 100
            sampling_z_step = 50
        
        tl_color_cap_db = 120
        
        # 转换网格坐标为实际距离
        enemy_y_km_vals = rows[:, 0] * sampling_y_step / 1000.0
        uuv_x_km_vals = rows[:, 1] * sampling_x_step / 1000.0
        uuv_y_km_vals = rows[:, 2] * sampling_y_step / 1000.0
        uuv_z_m_vals = rows[:, 3] * sampling_z_step
        tl_vals = rows[:, 4]
        
        # 将0 dB替换为限幅值
        tl_vals = np.where(tl_vals == 0, tl_color_cap_db, tl_vals)
        
        # 筛选与当前敌方Y坐标相关的数据（容差±0.5km）
        mask = np.abs(enemy_y_km_vals - enemy_y_km) <= 0.5
        print(f"敌方Y={enemy_y_km}, 筛选条件: ±0.5km, 符合点数: {np.sum(mask)}")
        
        result = {
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
        print(f"✅ TL数据加载成功, 有效点数: {result['stats']['valid_count']}")
        return result
    
    except Exception as e:
        print(f"❌ 加载TL数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# 测试
if __name__ == '__main__':
    print("=" * 60)
    print("测试后端函数")
    print("=" * 60)
    
    # 加载配置
    cfg = OmegaConf.load('configs/main_config.yaml')
    
    print("\n1️⃣ 测试 _load_terrain_data...")
    terrain = _load_terrain_data(None)
    
    print("\n2️⃣ 测试 _load_tl_data...")
    tl = _load_tl_data(cfg, 4.0)
    
    print("\n" + "=" * 60)
    print(f"结果: terrain={'✅' if terrain else '❌'}, tl={'✅' if tl else '❌'}")
    print("=" * 60)

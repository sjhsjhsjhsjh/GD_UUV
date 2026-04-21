#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Keypoints三维面板数据预处理脚本（Plotly版本）。

脚本功能：
1. 读取 configs/keypoint_dashboard.yaml 中的 dashboard 配置。
2. 加载 keypoints NPZ 与 terrain NPZ。
3. 将 key_pairs 还原为唯一 UUV 点集与敌方位置列表。
4. 生成前端直接加载的 JSON 文件：
   - payload.json
   - meta.json
   - config_runtime.json

运行示例：
    E:/lib/conda-env/torch_gpu/python.exe dashboards/keypoints_dashboard/prepare_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.rich_print import print_info, print_success  # noqa: E402



def _resolve_path(path_value: str | Path) -> Path:
    """
    将配置中的路径解析为绝对路径。

    输入参数：
        path_value: str | Path
            配置中的相对路径或绝对路径。

    输出参数：
        Path
            解析后的绝对路径。

    功能说明：
        如果路径是相对路径，则自动拼接项目根目录；如果是绝对路径则直接返回。

    调用示例：
        >>> _resolve_path('output/keypoints/key_points.npz')
    """
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj
    return PROJECT_ROOT / path_obj



def _load_dashboard_config(config_path: Path) -> DictConfig:
    """
    读取 keypoint 面板配置。

    输入参数：
        config_path: Path
            配置文件路径。

    输出参数：
        DictConfig
            OmegaConf 配置对象。

    功能说明：
        从 YAML 中读取 dashboard 节点，供后续数据预处理和运行时参数构建使用。

    调用示例：
        >>> cfg = _load_dashboard_config(PROJECT_ROOT / 'configs' / 'keypoint_dashboard.yaml')
    """
    cfg: DictConfig = OmegaConf.load(config_path)
    return cfg



def _build_payload(cfg: DictConfig, keypoint_npz: Any, terrain_npz: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    构建前端所需 payload、meta、runtime_config 三类数据对象。

    输入参数：
        cfg: DictConfig
            仪表盘配置对象。
        keypoint_npz: Any
            keypoints NPZ 对象，需包含 key_pairs、enemy_x、enemy_z。
        terrain_npz: Any
            terrain NPZ 对象，需包含 bathymetry_2d、x_coords、y_coords。

    输出参数：
        tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
            依次返回 payload、meta、runtime_config。

    功能说明：
        将点对数据压缩为“唯一 UUV 点集 + 敌方 y 轴离散列表”，
        保持与 TL 面板一致的数据组织习惯，同时显著降低前端负载。

    调用示例：
        >>> payload, meta, runtime_cfg = _build_payload(cfg, keypoint_npz, terrain_npz)
    """
    key_pairs = keypoint_npz['key_pairs'].astype(np.int32)
    enemy_x_m = int(keypoint_npz['enemy_x'])
    enemy_z_m = int(keypoint_npz['enemy_z'])

    terrain_depth_m = terrain_npz['bathymetry_2d'].astype(np.float32)
    terrain_x_km = terrain_npz['x_coords'].astype(np.float32)
    terrain_y_km = terrain_npz['y_coords'].astype(np.float32)

    enemy_y_values_m = np.unique(key_pairs[:, 0]).astype(np.int32)
    uuv_unique = np.unique(key_pairs[:, 1:4], axis=0).astype(np.int32)

    uuv_x_km = (uuv_unique[:, 0].astype(np.float32) * 0.001)
    uuv_y_km = (uuv_unique[:, 1].astype(np.float32) * 0.001)
    uuv_z_m = uuv_unique[:, 2].astype(np.float32)
    uuv_z_plot_m = -uuv_z_m

    point_color = uuv_z_m.copy()
    color_min_auto = float(np.min(point_color))
    color_max_auto = float(np.max(point_color))

    enemy_x_km = float(enemy_x_m) * 0.001
    enemy_positions: list[dict[str, Any]] = []
    for idx_value, enemy_y_m in enumerate(enemy_y_values_m.tolist()):
        enemy_y_km = float(enemy_y_m) * 0.001
        enemy_positions.append(
            {
                'index': idx_value,
                'enemy_x_km': enemy_x_km,
                'enemy_y_m': int(enemy_y_m),
                'enemy_y_km': enemy_y_km,
                'enemy_z_m': int(enemy_z_m),
                'label': f'敌方#{idx_value + 1} | x={enemy_x_km:.3f}km, y={enemy_y_km:.3f}km, z={enemy_z_m}m',
            }
        )

    points_per_enemy = int(key_pairs.shape[0] // enemy_y_values_m.shape[0])

    payload = {
        'metadata': {
            'total_key_pairs': int(key_pairs.shape[0]),
            'uuv_unique_points': int(uuv_unique.shape[0]),
            'enemy_positions_count': int(enemy_y_values_m.shape[0]),
            'points_per_enemy': points_per_enemy,
            'color_feature': 'uuv_z_m',
        },
        'terrain': {
            'x_km': terrain_x_km.tolist(),
            'y_km': terrain_y_km.tolist(),
            'depth_m': terrain_depth_m.tolist(),
            'shape': list(terrain_depth_m.shape),
        },
        'points': {
            'x_km': uuv_x_km.tolist(),
            'y_km': uuv_y_km.tolist(),
            'z_m': uuv_z_m.tolist(),
            'z_plot_m': uuv_z_plot_m.tolist(),
            'color_value': point_color.tolist(),
        },
        'enemy_positions': enemy_positions,
        'stats': {
            'enemy_x_km': enemy_x_km,
            'enemy_z_m': int(enemy_z_m),
            'color_min_auto': color_min_auto,
            'color_max_auto': color_max_auto,
            'terrain_depth_min_m': float(np.min(terrain_depth_m)),
            'terrain_depth_max_m': float(np.max(terrain_depth_m)),
            'uuv_x_min_km': float(np.min(uuv_x_km)),
            'uuv_x_max_km': float(np.max(uuv_x_km)),
            'uuv_y_min_km': float(np.min(uuv_y_km)),
            'uuv_y_max_km': float(np.max(uuv_y_km)),
            'uuv_z_min_m': float(np.min(uuv_z_m)),
            'uuv_z_max_m': float(np.max(uuv_z_m)),
            'initial_enemy_index': 0,
        },
    }

    runtime_config = {
        'color_min': None if cfg.dashboard.color_min is None else float(cfg.dashboard.color_min),
        'color_max': None if cfg.dashboard.color_max is None else float(cfg.dashboard.color_max),
        'camera_position': {
            'x': float(cfg.dashboard.camera_position.x),
            'y': float(cfg.dashboard.camera_position.y),
            'z': float(cfg.dashboard.camera_position.z),
        },
        'plotly_config': {
            'marker_size': float(cfg.dashboard.plotly_config.marker_size),
            'marker_opacity': float(cfg.dashboard.plotly_config.marker_opacity),
            'terrain_opacity': float(cfg.dashboard.plotly_config.terrain_opacity),
            'colorscale': str(cfg.dashboard.plotly_config.colorscale),
        },
    }

    meta = {
        'generated_by': 'prepare_data.py',
        'files': {
            'payload': 'payload.json',
            'runtime_config': 'config_runtime.json',
            'meta': 'meta.json',
        },
        'key_pair_count': int(key_pairs.shape[0]),
        'uuv_unique_count': int(uuv_unique.shape[0]),
        'enemy_count': int(enemy_y_values_m.shape[0]),
        'points_per_enemy': points_per_enemy,
    }

    return payload, meta, runtime_config



def main() -> int:
    """
    脚本主入口：读取配置、预处理数据并导出 JSON。

    输入参数：
        无。

    输出参数：
        int
            返回 0 表示执行完成。

    功能说明：
        串联完整流程：加载配置 -> 读取 NPZ -> 构建 payload/meta/runtime -> 写出 JSON。

    调用示例：
        >>> # E:/lib/conda-env/torch_gpu/python.exe dashboards/keypoints_dashboard/prepare_data.py
    """
    config_path = PROJECT_ROOT / 'configs' / 'keypoint_dashboard.yaml'
    cfg = _load_dashboard_config(config_path)

    keypoint_path = _resolve_path(cfg.dashboard.keypoint_npz_path)
    terrain_path = _resolve_path(cfg.dashboard.terrain_npz_path)
    output_dir = _resolve_path(cfg.dashboard.output_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_info(f'加载配置: {config_path}')
    print_info(f'关键点文件: {keypoint_path}')
    print_info(f'地形文件: {terrain_path}')
    print_info(f'输出目录: {output_dir}')

    with np.load(keypoint_path, allow_pickle=True) as keypoint_npz:
        with np.load(terrain_path, allow_pickle=True) as terrain_npz:
            payload, meta, runtime_cfg = _build_payload(cfg, keypoint_npz, terrain_npz)

    payload_path = output_dir / 'payload.json'
    meta_path = output_dir / 'meta.json'
    runtime_path = output_dir / 'config_runtime.json'

    payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
    runtime_path.write_text(json.dumps(runtime_cfg, ensure_ascii=False), encoding='utf-8')

    if bool(cfg.dashboard.verbose_stats):
        print_info(f"总点对数: {payload['metadata']['total_key_pairs']}")
        print_info(f"唯一UUV点数: {payload['metadata']['uuv_unique_points']}")
        print_info(f"敌方位置数: {payload['metadata']['enemy_positions_count']}")
        print_info(f"每个敌方对应点数: {payload['metadata']['points_per_enemy']}")

    print_success('Keypoints 面板数据预处理完成')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

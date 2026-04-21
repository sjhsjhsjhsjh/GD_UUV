#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TL三维传播损失面板数据预处理脚本（Plotly版本）。

脚本功能：
1. 读取 configs/TL_dashboard.yaml 中的 dashboard 配置；
2. 加载地形NPZ与TL缓存NPZ；
3. 按采样步长生成我方采样点；
4. 生成前端直接可加载的 JSON 文件：
   - payload.json
   - meta.json
   - config_runtime.json

运行示例：
    E:/lib/conda-env/torch_gpu/python.exe dashboards/TL_npz_dashboard/prepare_data.py
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

from utils.rich_print import print_error, print_info, print_success, print_warn  # noqa: E402


def _resolve_path(path_value: str | Path) -> Path:
    """
    将配置中的路径解析为绝对路径。

    参数：
        path_value: str | Path
            配置内路径，支持相对路径与绝对路径。

    返回：
        Path
            解析后的绝对路径对象。

    调用示例：
        >>> _resolve_path('TLdata/qianshui.npz')
    """
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj
    return PROJECT_ROOT / path_obj


def _load_dashboard_config(config_path: Path) -> DictConfig:
    """
    读取并校验仪表盘配置。

    参数：
        config_path: Path
            配置文件路径，通常为 configs/TL_dashboard.yaml。

    返回：
        DictConfig
            解析后的 OmegaConf 配置对象。

    异常：
        ValueError
            当配置缺少 dashboard 根节点时抛出。

    调用示例：
        >>> cfg = _load_dashboard_config(PROJECT_ROOT / 'configs' / 'TL_dashboard.yaml')
    """
    cfg_any = OmegaConf.load(config_path)
    if not isinstance(cfg_any, DictConfig):
        raise ValueError('配置根节点必须是字典结构')
    if 'dashboard' not in cfg_any:
        raise ValueError('配置文件缺少 dashboard 根节点')
    return cfg_any


def _load_main_config(config_path: Path) -> DictConfig:
    """
    读取主配置文件，用于提取敌方真实 x 坐标等环境语义。

    参数：
        config_path: Path
            主配置文件路径，通常为 configs/main_config.yaml。

    返回：
        DictConfig
            解析后的 OmegaConf 配置对象。

    异常：
        ValueError
            当配置缺少 env 根节点时抛出。

    调用示例：
        >>> cfg = _load_main_config(PROJECT_ROOT / 'configs' / 'main_config.yaml')
    """
    cfg_any = OmegaConf.load(config_path)
    if not isinstance(cfg_any, DictConfig):
        raise ValueError('主配置根节点必须是字典结构')
    if 'env' not in cfg_any:
        raise ValueError('主配置文件缺少 env 根节点')
    return cfg_any


def _nearest_index(axis_values: np.ndarray, target_value: float) -> int:
    """
    计算目标值在离散坐标轴上的最近邻索引。

    参数：
        axis_values: np.ndarray
            一维离散轴值数组。
        target_value: float
            连续输入值。

    返回：
        int
            最近邻索引。

    调用示例：
        >>> _nearest_index(np.array([0.0, 2.0, 4.0]), 2.9)
    """
    return int(np.argmin(np.abs(axis_values - target_value)))


def _validate_tl_shape(axes: dict[str, np.ndarray], tl_grid: np.ndarray, pair_mask: np.ndarray) -> None:
    """
    校验TL主网格与坐标轴长度一致性。

    参数：
        axes: dict[str, np.ndarray]
            坐标轴字典，必须包含 our_x_km、our_y_km、our_z_m、enemy_y_km、enemy_z_m。
        tl_grid: np.ndarray
            TL均值网格，预期形状为 (Nx, Ny, Nz, Ey, Ez)。
        pair_mask: np.ndarray
            有效掩码网格，形状需与 tl_grid 一致。

    返回：
        None
            校验通过时不返回内容。

    异常：
        ValueError
            当维度不匹配时抛出。

    调用示例：
        >>> _validate_tl_shape(axes, tl_grid, pair_mask)
    """
    expected_shape = (
        axes['our_x_km'].shape[0],
        axes['our_y_km'].shape[0],
        axes['our_z_m'].shape[0],
        axes['enemy_y_km'].shape[0],
        axes['enemy_z_m'].shape[0],
    )
    if tl_grid.shape != expected_shape:
        raise ValueError(f'tl_mean_grid 形状不匹配，期望 {expected_shape}，实际 {tl_grid.shape}')
    if pair_mask.shape != expected_shape:
        raise ValueError(f'pair_mask 形状不匹配，期望 {expected_shape}，实际 {pair_mask.shape}')


def _build_sampled_indices(nx: int, ny: int, nz: int, step_x: int, step_y: int, step_z: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    根据采样步长生成我方三维索引。

    参数：
        nx: int
            x轴长度。
        ny: int
            y轴长度。
        nz: int
            z轴长度。
        step_x: int
            x轴采样步长，>=1。
        step_y: int
            y轴采样步长，>=1。
        step_z: int
            z轴采样步长，>=1。

    返回：
        tuple[np.ndarray, np.ndarray, np.ndarray]
            三个等长一维索引数组 (ix, iy, iz)。

    调用示例：
        >>> ix, iy, iz = _build_sampled_indices(4, 3, 2, 1, 1, 1)
    """
    x_idx = np.arange(0, nx, step_x, dtype=np.int32)
    y_idx = np.arange(0, ny, step_y, dtype=np.int32)
    z_idx = np.arange(0, nz, step_z, dtype=np.int32)
    gx, gy, gz = np.meshgrid(x_idx, y_idx, z_idx, indexing='ij')
    return gx.ravel(), gy.ravel(), gz.ravel()


def _to_json_float_list(values: np.ndarray) -> list[float | None]:
    """
    将 numpy 数组安全转换为 JSON 可序列化浮点列表。

    参数：
        values: np.ndarray
            任意形状浮点数组。

    返回：
        list[float | None]
            有限值转为 float，非有限值转为 None。

    调用示例：
        >>> _to_json_float_list(np.array([1.0, np.nan]))
    """
    result: list[float | None] = []
    flat = values.ravel()
    for value in flat:
        if np.isfinite(value):
            result.append(float(value))
        else:
            result.append(None)
    return result


def _build_payload(cfg: DictConfig, main_cfg: DictConfig, terrain_npz: Any, tl_npz: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    构建前端所需 payload/meta/runtime 三类JSON对象。

    参数：
        cfg: DictConfig
            仪表盘配置对象。
        terrain_npz: Any
            地形 NPZ 对象，需包含 bathymetry_2d、x_coords、y_coords。
            main_cfg: DictConfig
                主配置对象，需包含 env.enemy_x。
        tl_npz: Any
            TL NPZ 对象，需包含坐标轴、tl_mean_grid、pair_mask。

    返回：
        tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
            payload、meta、runtime_config 三个字典。

    调用示例：
        >>> payload, meta, runtime_cfg = _build_payload(cfg, terrain_npz, tl_npz)
    """
    terrain_depth = terrain_npz['bathymetry_2d'].astype(np.float32)
    terrain_x = terrain_npz['x_coords'].astype(np.float32)
    terrain_y = terrain_npz['y_coords'].astype(np.float32)
    enemy_x_km = float(main_cfg.env.enemy_x) * 0.001

    axes = {
        'our_x_km': tl_npz['our_x_km'].astype(np.float32),
        'our_y_km': tl_npz['our_y_km'].astype(np.float32),
        'our_z_m': tl_npz['our_z_m'].astype(np.float32),
        'enemy_y_km': tl_npz['enemy_y_km'].astype(np.float32),
        'enemy_z_m': tl_npz['enemy_z_m'].astype(np.float32),
    }
    tl_grid = tl_npz['tl_mean_grid'].astype(np.float32)
    pair_mask = tl_npz['pair_mask'].astype(bool)
    _validate_tl_shape(axes, tl_grid, pair_mask)

    step_x = max(1, int(cfg.dashboard.sample_step_x))
    step_y = max(1, int(cfg.dashboard.sample_step_y))
    step_z = max(1, int(cfg.dashboard.sample_step_z))
    use_pair_mask = bool(cfg.dashboard.use_pair_mask)
    nan_policy = str(cfg.dashboard.nan_policy).lower()
    if nan_policy not in {'drop', 'keep'}:
        raise ValueError('nan_policy 仅支持 drop 或 keep')

    ix, iy, iz = _build_sampled_indices(
        axes['our_x_km'].shape[0],
        axes['our_y_km'].shape[0],
        axes['our_z_m'].shape[0],
        step_x,
        step_y,
        step_z,
    )

    sampled_points_x = axes['our_x_km'][ix]
    sampled_points_y = axes['our_y_km'][iy]
    sampled_points_z = axes['our_z_m'][iz]

    sampled_tl = tl_grid[ix, iy, iz, :, :].transpose(1, 2, 0).astype(np.float32)
    sampled_mask = pair_mask[ix, iy, iz, :, :].transpose(1, 2, 0)

    if not use_pair_mask:
        sampled_mask = np.ones_like(sampled_mask, dtype=bool)

    finite_mask = np.isfinite(sampled_tl)
    if nan_policy == 'drop':
        sampled_mask = sampled_mask & finite_mask

    enemy_positions: list[dict[str, Any]] = []
    tl_by_enemy: list[dict[str, Any]] = []
    valid_values_for_global: list[np.ndarray] = []

    enemy_index = 0
    for enemy_y_idx in range(axes['enemy_y_km'].shape[0]):
        for enemy_z_idx in range(axes['enemy_z_m'].shape[0]):
            enemy_y_val = float(axes['enemy_y_km'][enemy_y_idx])
            enemy_z_val = float(axes['enemy_z_m'][enemy_z_idx])

            enemy_positions.append(
                {
                    'index': enemy_index,
                    'enemy_x_km': enemy_x_km,
                    'enemy_y_km': enemy_y_val,
                    'enemy_z_m': enemy_z_val,
                    'label': f'敌方#{enemy_index + 1} | x={enemy_x_km:.3f}km, y={enemy_y_val:.3f}km, z={enemy_z_val:.3f}m',
                    'enemy_y_idx': enemy_y_idx,
                    'enemy_z_idx': enemy_z_idx,
                }
            )

            slice_values = sampled_tl[enemy_y_idx, enemy_z_idx, :]
            slice_mask = sampled_mask[enemy_y_idx, enemy_z_idx, :]

            valid_values = slice_values[slice_mask & np.isfinite(slice_values)]
            if valid_values.size > 0:
                valid_values_for_global.append(valid_values)

            tl_by_enemy.append(
                {
                    'enemy_index': enemy_index,
                    'enemy_y_km': enemy_y_val,
                    'enemy_z_m': enemy_z_val,
                    'tl_values': _to_json_float_list(slice_values),
                    'valid_mask': slice_mask.astype(np.uint8).tolist(),
                }
            )
            enemy_index += 1

    if valid_values_for_global:
        all_valid = np.concatenate(valid_values_for_global)
        global_tl_min = float(np.min(all_valid))
        global_tl_max = float(np.max(all_valid))
    else:
        global_tl_min = float('nan')
        global_tl_max = float('nan')
        print_warn('所有敌方切片都没有有效TL值，请检查输入数据')

    init_enemy_y_idx = _nearest_index(axes['enemy_y_km'], float(cfg.dashboard.initial_enemy_y_km))
    init_enemy_z_idx = _nearest_index(axes['enemy_z_m'], float(cfg.dashboard.initial_enemy_z_m))
    init_enemy_flat = init_enemy_y_idx * axes['enemy_z_m'].shape[0] + init_enemy_z_idx

    init_values = sampled_tl[init_enemy_y_idx, init_enemy_z_idx, :]
    init_mask = sampled_mask[init_enemy_y_idx, init_enemy_z_idx, :]
    init_valid_values = init_values[init_mask & np.isfinite(init_values)]

    payload = {
        'metadata': {
            'our_points_per_enemy': int(sampled_points_x.shape[0]),
            'enemy_positions_count': int(len(enemy_positions)),
            'total_data_points': int(sampled_points_x.shape[0] * len(enemy_positions)),
            'sample_steps': {'x': step_x, 'y': step_y, 'z': step_z},
            'pair_mask_used': use_pair_mask,
            'nan_policy': nan_policy,
        },
        'axes': {
            'enemy_x_km': [enemy_x_km],
            'our_x_km': axes['our_x_km'].tolist(),
            'our_y_km': axes['our_y_km'].tolist(),
            'our_z_m': axes['our_z_m'].tolist(),
            'enemy_y_km': axes['enemy_y_km'].tolist(),
            'enemy_z_m': axes['enemy_z_m'].tolist(),
        },
        'enemy_positions': enemy_positions,
        'terrain': {
            'x_km': terrain_x.tolist(),
            'y_km': terrain_y.tolist(),
            'depth_m': terrain_depth.tolist(),
            'shape': list(terrain_depth.shape),
        },
        'points': {
            'x_km': sampled_points_x.astype(np.float32).tolist(),
            'y_km': sampled_points_y.astype(np.float32).tolist(),
            'z_m': sampled_points_z.astype(np.float32).tolist(),
        },
        'tl_by_enemy': tl_by_enemy,
        'stats': {
            'tl_min_db': global_tl_min,
            'tl_max_db': global_tl_max,
            'enemy_count': int(len(enemy_positions)),
            'points_per_enemy': int(sampled_points_x.shape[0]),
            'enemy_x_km': enemy_x_km,
            'terrain_depth_min_m': float(np.min(terrain_depth)),
            'terrain_depth_max_m': float(np.max(terrain_depth)),
            'initial_enemy_index': int(init_enemy_flat),
        },
    }

    runtime_config = {
        'tl_color_min_db': None if cfg.dashboard.tl_color_min_db is None else float(cfg.dashboard.tl_color_min_db),
        'tl_color_max_db': None if cfg.dashboard.tl_color_max_db is None else float(cfg.dashboard.tl_color_max_db),
        'tl_color_cap_db': None if cfg.dashboard.tl_color_cap_db is None else float(cfg.dashboard.tl_color_cap_db),
        'z_visual_scale': float(cfg.dashboard.z_visual_scale),
        'terrain_z_exaggeration': float(cfg.dashboard.terrain_z_exaggeration),
        'camera_position': {
            'x': float(cfg.dashboard.camera_position.x),
            'y': float(cfg.dashboard.camera_position.y),
            'z': float(cfg.dashboard.camera_position.z),
        },
        'background_top_color': str(cfg.dashboard.background_top_color),
        'background_bottom_color': str(cfg.dashboard.background_bottom_color),
        'plotly_config': {
            'marker_size': float(cfg.dashboard.plotly_config.marker_size),
            'marker_opacity': float(cfg.dashboard.plotly_config.marker_opacity),
            'terrain_marker_size': float(cfg.dashboard.plotly_config.terrain_marker_size),
            'terrain_opacity': float(cfg.dashboard.plotly_config.terrain_opacity),
            'colorscale': str(cfg.dashboard.plotly_config.colorscale),
        },
        'initial_enemy': {
            'input_enemy_y_km': float(cfg.dashboard.initial_enemy_y_km),
            'input_enemy_z_m': float(cfg.dashboard.initial_enemy_z_m),
            'mapped_enemy_x_km': enemy_x_km,
            'mapped_enemy_y_idx': int(init_enemy_y_idx),
            'mapped_enemy_z_idx': int(init_enemy_z_idx),
            'mapped_enemy_index': int(init_enemy_flat),
            'mapped_enemy_x_km': enemy_x_km,
            'mapped_enemy_y_km': float(axes['enemy_y_km'][init_enemy_y_idx]),
            'mapped_enemy_z_m': float(axes['enemy_z_m'][init_enemy_z_idx]),
        },
    }

    meta = {
        'generated_by': 'prepare_data.py',
        'files': {
            'payload': 'payload.json',
            'runtime_config': 'config_runtime.json',
        },
        'axes_lengths': {
            'our_x': int(axes['our_x_km'].shape[0]),
            'our_y': int(axes['our_y_km'].shape[0]),
            'our_z': int(axes['our_z_m'].shape[0]),
            'enemy_y': int(axes['enemy_y_km'].shape[0]),
            'enemy_z': int(axes['enemy_z_m'].shape[0]),
        },
        'sampled_points': int(sampled_points_x.shape[0]),
        'enemy_positions': int(len(enemy_positions)),
        'initial_slice_stats': {
            'valid_count': int(init_valid_values.size),
            'tl_min_db': float(np.min(init_valid_values)) if init_valid_values.size > 0 else None,
            'tl_max_db': float(np.max(init_valid_values)) if init_valid_values.size > 0 else None,
            'tl_mean_db': float(np.mean(init_valid_values)) if init_valid_values.size > 0 else None,
        },
    }

    return payload, meta, runtime_config


def main() -> int:
    """
    脚本主入口：执行配置读取、数据预处理与JSON导出。

    参数：
        无。

    返回：
        int
            0 表示成功，1 表示失败。

    调用示例：
        >>> # E:/lib/conda-env/torch_gpu/python.exe dashboards/TL_npz_dashboard/prepare_data.py
    """
    try:
        config_path = PROJECT_ROOT / 'configs' / 'TL_dashboard.yaml'
        main_config_path = PROJECT_ROOT / 'configs' / 'main_config.yaml'
        cfg = _load_dashboard_config(config_path)
        main_cfg = _load_main_config(main_config_path)

        terrain_path = _resolve_path(cfg.dashboard.terrain_npz_path)
        tl_cache_path = _resolve_path(cfg.dashboard.tl_cache_npz_path)
        output_dir = _resolve_path(cfg.dashboard.output_data_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print_info(f'加载配置: {config_path}')
        print_info(f'主配置: {main_config_path}')
        print_info(f'地形文件: {terrain_path}')
        print_info(f'TL缓存文件: {tl_cache_path}')
        print_info(f'输出目录: {output_dir}')

        if not terrain_path.exists():
            raise FileNotFoundError(f'地形文件不存在: {terrain_path}')
        if not tl_cache_path.exists():
            raise FileNotFoundError(f'TL缓存文件不存在: {tl_cache_path}')

        with np.load(terrain_path, allow_pickle=True) as terrain_npz:
            with np.load(tl_cache_path, allow_pickle=True) as tl_npz:
                payload, meta, runtime_cfg = _build_payload(cfg, main_cfg, terrain_npz, tl_npz)

        payload_path = output_dir / 'payload.json'
        meta_path = output_dir / 'meta.json'
        runtime_path = output_dir / 'config_runtime.json'

        payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
        runtime_path.write_text(json.dumps(runtime_cfg, ensure_ascii=False), encoding='utf-8')

        if bool(cfg.dashboard.verbose_stats):
            print_info(f"敌方位置数量: {meta['enemy_positions']}")
            print_info(f"每个敌方位置对应我方采样点: {meta['sampled_points']}")
            init_stats = meta['initial_slice_stats']
            print_info(
                f"初始切片有效点: {init_stats['valid_count']} | "
                f"TL范围(dB): {init_stats['tl_min_db']} ~ {init_stats['tl_max_db']}"
            )

        print_success('TL面板数据预处理完成（Plotly版本）')
        return 0
    except Exception as exc:
        print_error(f'预处理失败: {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())


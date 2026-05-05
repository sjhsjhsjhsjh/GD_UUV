#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Final TL 三维传播损失面板数据预处理脚本（Plotly版本）。

脚本功能：
1. 读取 configs/final_tl_dashboard.yaml 中的 dashboard 配置；
2. 读取地形 txt 与 TL csv/txt（行格式: enemy_y,uuv_x,uuv_y,uuz_z,tl）；
3. 构建与 TL_npz_dashboard 完全一致的 payload/meta/runtime JSON 结构；
4. 输出到 dashboards/final_tl_dashboard/data 目录供前端直接读取。

运行示例：
    E:/lib/conda-env/torch_gpu/python.exe dashboards/final_tl_dashboard/prepare_data.py
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

    输入参数：
        path_value: str | Path
            配置内路径，支持相对路径与绝对路径。

    输出参数：
        Path
            解析后的绝对路径对象。

    功能说明：
        若输入为相对路径，则默认以项目根目录为基准拼接。

    调用示例：
        >>> _resolve_path('TLdata/final_tl_data.txt')
    """
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj
    return PROJECT_ROOT / path_obj


def _load_dashboard_config(config_path: Path) -> DictConfig:
    """
    读取并校验 final_tl_dashboard 配置。

    输入参数：
        config_path: Path
            配置文件路径，通常为 configs/final_tl_dashboard.yaml。

    输出参数：
        DictConfig
            解析后的 OmegaConf 配置对象。

    功能说明：
        该函数统一检查 dashboard 根节点存在性，保证后续字段读取稳定。

    调用示例：
        >>> cfg = _load_dashboard_config(PROJECT_ROOT / 'configs' / 'final_tl_dashboard.yaml')
    """
    cfg_any = OmegaConf.load(config_path)
    if not isinstance(cfg_any, DictConfig):
        raise ValueError('配置根节点必须是字典结构')
    if 'dashboard' not in cfg_any:
        raise ValueError('配置文件缺少 dashboard 根节点')
    return cfg_any


def _load_main_config(config_path: Path) -> DictConfig:
    """
    读取主配置文件，用于获取敌方固定坐标和地形网格参数。

    输入参数：
        config_path: Path
            主配置文件路径，通常为 configs/main_config.yaml。

    输出参数：
        DictConfig
            解析后的 OmegaConf 配置对象。

    功能说明：
        final_tl_dashboard 约定敌方 x/z 固定，均从主配置 env 节点读取。

    调用示例：
        >>> main_cfg = _load_main_config(PROJECT_ROOT / 'configs' / 'main_config.yaml')
    """
    cfg_any = OmegaConf.load(config_path)
    if not isinstance(cfg_any, DictConfig):
        raise ValueError('主配置根节点必须是字典结构')
    if 'env' not in cfg_any:
        raise ValueError('主配置缺少 env 根节点')
    return cfg_any


def _to_json_float_list(values: np.ndarray) -> list[float | None]:
    """
    将 numpy 浮点数组转换为 JSON 可序列化列表。

    输入参数：
        values: np.ndarray
            任意形状浮点数组。

    输出参数：
        list[float | None]
            有限值写入 float，非有限值写入 None。

    功能说明：
        与 TL_npz_dashboard 保持一致，前端用 null 识别无效数据。

    调用示例：
        >>> _to_json_float_list(np.array([1.0, np.nan], dtype=np.float32))
    """
    result: list[float | None] = []
    for value in values.ravel():
        if np.isfinite(value):
            result.append(float(value))
        else:
            result.append(None)
    return result


def _nearest_index(axis_values: np.ndarray, target_value: float) -> int:
    """
    计算目标值在离散轴上的最近邻索引。

    输入参数：
        axis_values: np.ndarray
            一维坐标轴数组。
        target_value: float
            目标连续值。

    输出参数：
        int
            最近邻索引。

    功能说明：
        用于将运行时初始敌方 y/z 连续值映射到离散索引。

    调用示例：
        >>> _nearest_index(np.array([1.0, 3.0, 5.0], dtype=np.float32), 3.2)
    """
    return int(np.argmin(np.abs(axis_values - target_value)))


def _parse_numeric_rows(text_path: Path) -> list[list[float]]:
    """
    从文本文件中读取浮点行并自动跳过表头。

    输入参数：
        text_path: Path
            文本路径，支持逗号或空白分隔。

    输出参数：
        list[list[float]]
            逐行浮点值列表。

    功能说明：
        该函数用于兼容 csv/txt 两类文本输入。
        当行内存在非数值（如表头 enemy_y,uuv_x,...）时自动忽略。

    调用示例：
        >>> rows = _parse_numeric_rows(PROJECT_ROOT / 'TLdata' / 'final_tl_data.txt')
    """
    numeric_rows: list[list[float]] = []
    raw_text = text_path.read_text(encoding='utf-8')
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        tokens = line.replace(',', ' ').split()
        if not tokens:
            continue
        try:
            row_values = [float(token) for token in tokens]
            numeric_rows.append(row_values)
        except ValueError:
            continue
    return numeric_rows


def _load_terrain_npz(terrain_npz_path: Path) -> dict[str, Any]:
    """
    从 NPZ 文件读取地形数据并转换为前端地形网格。

    输入参数：
        terrain_npz_path: Path
            地形 NPZ 文件路径，应包含 bathymetry_2d、x_coords、y_coords 数据。

    输出参数：
        dict[str, Any]
            字段包含 x_km、y_km、depth_m、shape。

    功能说明：
        从 NPZ 文件提取海底深度数据与坐标轴，构建前端所需的地形网格结构。
        - bathymetry_2d: 形状 (ny, nx)，海底深度值（米）
        - x_coords: 形状 (nx,)，X 坐标轴（km）
        - y_coords: 形状 (ny,)，Y 坐标轴（km）

    调用示例：
        >>> terrain = _load_terrain_npz(PROJECT_ROOT / 'output' / 'bty' / 'terrain.npz')
    """
    try:
        data = np.load(terrain_npz_path)
    except FileNotFoundError:
        raise FileNotFoundError(f'地形 NPZ 文件不存在: {terrain_npz_path}')
    except Exception as exc:
        raise ValueError(f'无法加载 NPZ 文件 {terrain_npz_path}: {exc}')

    # 验证必需字段
    required_keys = {'bathymetry_2d', 'x_coords', 'y_coords'}
    missing_keys = required_keys - set(data.files)
    if missing_keys:
        raise ValueError(
            f'NPZ 文件缺少必需字段: {missing_keys}. '
            f'存在字段: {set(data.files)}. '
            f'文件: {terrain_npz_path}'
        )

    # 提取数据
    bathymetry_2d = data['bathymetry_2d'].astype(np.float32)
    x_coords = data['x_coords'].astype(np.float32)
    y_coords = data['y_coords'].astype(np.float32)

    # 验证形状匹配
    ny, nx = bathymetry_2d.shape
    if x_coords.shape[0] != nx:
        raise ValueError(
            f'坐标轴长度不匹配: x_coords 长度 {x_coords.shape[0]} '
            f'不等于 bathymetry_2d 宽度 {nx}. 文件: {terrain_npz_path}'
        )
    if y_coords.shape[0] != ny:
        raise ValueError(
            f'坐标轴长度不匹配: y_coords 长度 {y_coords.shape[0]} '
            f'不等于 bathymetry_2d 高度 {ny}. 文件: {terrain_npz_path}'
        )

    return {
        'x_km': x_coords.tolist(),
        'y_km': y_coords.tolist(),
        'depth_m': bathymetry_2d.tolist(),
        'shape': [int(ny), int(nx)],
    }


def _load_tl_csv_rows(tl_csv_path: Path, main_cfg: DictConfig, cfg: DictConfig) -> np.ndarray:
    """
    读取 TL 文本数据、转换网格坐标为实际距离，并返回标准五列数组。

    输入参数：
        tl_csv_path: Path
            TL 文本路径，每行格式 grid_enemy_y,grid_uuv_x,grid_uuv_y,grid_uuv_z,tl。
            * 网格坐标（整数或浮点）表示相对于采样步长的网格位置。
        main_cfg: DictConfig
            主配置对象，提供 sampling_x_step、sampling_y_step、sampling_z_step 转换参数。
        cfg: DictConfig
            dashboard 配置对象，提供 tl_color_cap_db（用于替换 0 dB 值）。

    输出参数：
        np.ndarray
            形状为 (N, 5) 的 float32 数组，列顺序为 enemy_y,uuv_x,uuv_y,uuv_z,tl。
            * 转换后单位： enemy_y/uuv_x/uuv_y 为 km，uuv_z 为 m，tl 为 dB。

    功能说明：
        将输入的网格坐标转换为实际距离：
        1. enemy_y（网格） × sampling_y_step / 1000 → enemy_y（km）
        2. uuv_x（网格） × sampling_x_step / 1000 → uuv_x（km）
        3. uuv_y（网格） × sampling_y_step / 1000 → uuv_y（km）
        4. uuv_z（网格） × sampling_z_step → uuv_z（m）
        5. tl（dB）值为 0 时，替换为 tl_color_cap_db；否则保持不变。

    调用示例：
        >>> cfg = _load_dashboard_config(PROJECT_ROOT / 'configs' / 'final_tl_dashboard.yaml')
        >>> main_cfg = _load_main_config(PROJECT_ROOT / 'configs' / 'main_config.yaml')
        >>> tl_rows = _load_tl_csv_rows(PROJECT_ROOT / 'TLdata' / 'average_TL_results.txt', main_cfg, cfg)
    """

    rows = _parse_numeric_rows(tl_csv_path)
    if not rows:
        raise ValueError(f'TL 文件没有可用数值行: {tl_csv_path}')

    # 从配置中读取采样步长（单位：米）
    sampling_x_step = float(main_cfg.env.sampling_x_step)
    sampling_y_step = float(main_cfg.env.sampling_y_step)
    sampling_z_step = float(main_cfg.env.sampling_z_step)
    
    # 从 dashboard 配置中读取 TL 最大损失值，用于替换 0 dB
    tl_cap = float(cfg.dashboard.tl_color_cap_db)

    selected_rows: list[list[float]] = []
    for row in rows:
        if len(row) < 5:
            continue
        
        # 提取前5列
        grid_enemy_y = float(row[0])
        grid_uuv_x = float(row[1])
        grid_uuv_y = float(row[2])
        grid_uuv_z = float(row[3])
        tl_value = float(row[4])
        
        # 转换网格坐标为实际距离
        # 水平坐标（X、Y）转为 km，垂直坐标（Z）转为 m
        actual_enemy_y = grid_enemy_y * sampling_y_step / 1000.0
        actual_uuv_x = grid_uuv_x * sampling_x_step / 1000.0
        actual_uuv_y = grid_uuv_y * sampling_y_step / 1000.0
        actual_uuv_z = grid_uuv_z * sampling_z_step
        
        # 如果 TL 值为 0，替换为最大损失值
        if tl_value == 0.0:
            tl_value = tl_cap
        
        selected_rows.append([actual_enemy_y, actual_uuv_x, actual_uuv_y, actual_uuv_z, tl_value])

    if not selected_rows:
        raise ValueError(f'TL 文件缺少有效五列数据: {tl_csv_path}')

    return np.asarray(selected_rows, dtype=np.float32)


def _build_payload(
    cfg: DictConfig,
    main_cfg: DictConfig,
    terrain: dict[str, Any],
    tl_rows: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    将 CSV 行数据转换为前端所需 payload/meta/runtime 三类结构。

    输入参数：
        cfg: DictConfig
            final_tl_dashboard 配置对象。
        main_cfg: DictConfig
            主配置对象，提供敌方固定坐标与网格语义。
        terrain: dict[str, Any]
            地形网格对象，包含 x_km/y_km/depth_m/shape。
        tl_rows: np.ndarray
            TL 行数据，列顺序为 enemy_y,uuv_x,uuv_y,uuv_z,tl。

    输出参数：
        tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
            依次为 payload、meta、runtime_config。

    功能说明：
        该函数保证输出字段与 TL_npz_dashboard 完全一致，使前端逻辑无需改动。

    调用示例：
        >>> payload, meta, runtime_cfg = _build_payload(cfg, main_cfg, terrain, tl_rows)
    """
    enemy_y_all = tl_rows[:, 0].astype(np.float32)
    uuv_x_all = tl_rows[:, 1].astype(np.float32)
    uuv_y_all = tl_rows[:, 2].astype(np.float32)
    uuv_z_all = tl_rows[:, 3].astype(np.float32)
    tl_all = tl_rows[:, 4].astype(np.float32)

    unique_enemy_y = np.unique(np.round(enemy_y_all, 6)).astype(np.float32)
    unique_uuv_x = np.unique(np.round(uuv_x_all, 6)).astype(np.float32)
    unique_uuv_y = np.unique(np.round(uuv_y_all, 6)).astype(np.float32)
    unique_uuv_z = np.unique(np.round(uuv_z_all, 6)).astype(np.float32)

    step_x = max(1, int(cfg.dashboard.sample_step_x))
    step_y = max(1, int(cfg.dashboard.sample_step_y))
    step_z = max(1, int(cfg.dashboard.sample_step_z))

    sampled_x = unique_uuv_x[::step_x]
    sampled_y = unique_uuv_y[::step_y]
    sampled_z = unique_uuv_z[::step_z]

    gx, gy, gz = np.meshgrid(sampled_x, sampled_y, sampled_z, indexing='ij')
    point_x = gx.ravel().astype(np.float32)
    point_y = gy.ravel().astype(np.float32)
    point_z = gz.ravel().astype(np.float32)

    point_key_to_index: dict[tuple[float, float, float], int] = {}
    for idx in range(point_x.shape[0]):
        key = (float(np.round(point_x[idx], 6)), float(np.round(point_y[idx], 6)), float(np.round(point_z[idx], 6)))
        point_key_to_index[key] = idx

    enemy_x_km = float(main_cfg.env.enemy_x) * 0.001
    enemy_z_m = float(main_cfg.env.enemy_z)
    enemy_z_axis = np.asarray([enemy_z_m], dtype=np.float32)

    nan_policy = str(cfg.dashboard.nan_policy).lower()
    if nan_policy not in {'drop', 'keep'}:
        raise ValueError('nan_policy 仅支持 drop 或 keep')
    use_pair_mask = bool(cfg.dashboard.use_pair_mask)

    grouped_indices: dict[float, list[int]] = {float(value): [] for value in unique_enemy_y}
    for row_idx in range(tl_rows.shape[0]):
        key_enemy_y = float(np.round(enemy_y_all[row_idx], 6))
        grouped_indices[key_enemy_y].append(row_idx)

    enemy_positions: list[dict[str, Any]] = []
    tl_by_enemy: list[dict[str, Any]] = []
    valid_values_for_global: list[np.ndarray] = []

    for enemy_index, enemy_y_value in enumerate(unique_enemy_y):
        enemy_y_float = float(enemy_y_value)
        values = np.full(point_x.shape[0], np.nan, dtype=np.float32)
        valid_mask = np.zeros(point_x.shape[0], dtype=bool)

        for row_idx in grouped_indices[enemy_y_float]:
            point_key = (
                float(np.round(uuv_x_all[row_idx], 6)),
                float(np.round(uuv_y_all[row_idx], 6)),
                float(np.round(uuv_z_all[row_idx], 6)),
            )
            mapped_idx = point_key_to_index.get(point_key)
            if mapped_idx is None:
                continue
            values[mapped_idx] = float(tl_all[row_idx])
            valid_mask[mapped_idx] = True

        if not use_pair_mask:
            valid_mask = np.ones_like(valid_mask, dtype=bool)

        finite_mask = np.isfinite(values)
        if nan_policy == 'drop':
            valid_mask = valid_mask & finite_mask

        valid_values = values[valid_mask & finite_mask]
        if valid_values.size > 0:
            valid_values_for_global.append(valid_values)

        enemy_positions.append(
            {
                'index': int(enemy_index),
                'enemy_x_km': enemy_x_km,
                'enemy_y_km': enemy_y_float,
                'enemy_z_m': enemy_z_m,
                'label': f'敌方#{enemy_index + 1} | x={enemy_x_km:.3f}km, y={enemy_y_float:.3f}km, z={enemy_z_m:.3f}m',
                'enemy_y_idx': int(enemy_index),
                'enemy_z_idx': 0,
            }
        )

        tl_by_enemy.append(
            {
                'enemy_index': int(enemy_index),
                'enemy_y_km': enemy_y_float,
                'enemy_z_m': enemy_z_m,
                'tl_values': _to_json_float_list(values),
                'valid_mask': valid_mask.astype(np.uint8).tolist(),
            }
        )

    if valid_values_for_global:
        all_valid = np.concatenate(valid_values_for_global)
        global_tl_min = float(np.min(all_valid))
        global_tl_max = float(np.max(all_valid))
    else:
        global_tl_min = float('nan')
        global_tl_max = float('nan')
        print_warn('所有敌方切片均无有效 TL 值，请检查输入数据')

    init_enemy_y_idx = _nearest_index(unique_enemy_y, float(cfg.dashboard.initial_enemy_y_km))
    init_enemy_z_idx = _nearest_index(enemy_z_axis, float(cfg.dashboard.initial_enemy_z_m))
    init_enemy_flat = int(init_enemy_y_idx * enemy_z_axis.shape[0] + init_enemy_z_idx)

    init_values = np.asarray(tl_by_enemy[init_enemy_flat]['tl_values'], dtype=np.float32)
    init_valid_mask = np.asarray(tl_by_enemy[init_enemy_flat]['valid_mask'], dtype=np.uint8) > 0
    init_valid_values = init_values[init_valid_mask & np.isfinite(init_values)]

    terrain_depth = np.asarray(terrain['depth_m'], dtype=np.float32)

    payload = {
        'metadata': {
            'our_points_per_enemy': int(point_x.shape[0]),
            'enemy_positions_count': int(len(enemy_positions)),
            'total_data_points': int(point_x.shape[0] * len(enemy_positions)),
            'sample_steps': {'x': step_x, 'y': step_y, 'z': step_z},
            'pair_mask_used': use_pair_mask,
            'nan_policy': nan_policy,
        },
        'axes': {
            'enemy_x_km': [enemy_x_km],
            'our_x_km': unique_uuv_x.tolist(),
            'our_y_km': unique_uuv_y.tolist(),
            'our_z_m': unique_uuv_z.tolist(),
            'enemy_y_km': unique_enemy_y.tolist(),
            'enemy_z_m': enemy_z_axis.tolist(),
        },
        'enemy_positions': enemy_positions,
        'terrain': {
            'x_km': terrain['x_km'],
            'y_km': terrain['y_km'],
            'depth_m': terrain['depth_m'],
            'shape': terrain['shape'],
        },
        'points': {
            'x_km': point_x.tolist(),
            'y_km': point_y.tolist(),
            'z_m': point_z.tolist(),
        },
        'tl_by_enemy': tl_by_enemy,
        'stats': {
            'tl_min_db': global_tl_min,
            'tl_max_db': global_tl_max,
            'enemy_count': int(len(enemy_positions)),
            'points_per_enemy': int(point_x.shape[0]),
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
            'mapped_enemy_y_km': float(unique_enemy_y[init_enemy_y_idx]),
            'mapped_enemy_z_m': enemy_z_m,
        },
    }

    meta = {
        'generated_by': 'prepare_data.py',
        'files': {
            'payload': 'payload.json',
            'runtime_config': 'config_runtime.json',
        },
        'axes_lengths': {
            'our_x': int(unique_uuv_x.shape[0]),
            'our_y': int(unique_uuv_y.shape[0]),
            'our_z': int(unique_uuv_z.shape[0]),
            'enemy_y': int(unique_enemy_y.shape[0]),
            'enemy_z': int(enemy_z_axis.shape[0]),
        },
        'sampled_points': int(point_x.shape[0]),
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
    脚本主入口：执行配置读取、数据转换与 JSON 导出。

    输入参数：
        无。

    输出参数：
        int
            0 表示成功，1 表示失败。

    功能说明：
        该函数固定读取 final_tl_dashboard 配置，并将处理结果写入对应 data 目录。

    调用示例：
        >>> # E:/lib/conda-env/torch_gpu/python.exe dashboards/final_tl_dashboard/prepare_data.py
    """
    try:
        config_path = PROJECT_ROOT / 'configs' / 'final_tl_dashboard.yaml'
        main_config_path = PROJECT_ROOT / 'configs' / 'main_config.yaml'

        cfg = _load_dashboard_config(config_path)
        main_cfg = _load_main_config(main_config_path)

        terrain_txt_path = _resolve_path(cfg.dashboard.terrain_txt_path)
        tl_csv_path = _resolve_path(cfg.dashboard.tl_csv_path)
        output_dir = _resolve_path(cfg.dashboard.output_data_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print_info(f'加载配置: {config_path}')
        print_info(f'主配置: {main_config_path}')
        print_info(f'地形文本: {terrain_txt_path}')
        print_info(f'TL文本: {tl_csv_path}')
        print_info(f'输出目录: {output_dir}')

        if not terrain_txt_path.exists():
            raise FileNotFoundError(f'地形文件不存在: {terrain_txt_path}')
        if not tl_csv_path.exists():
            raise FileNotFoundError(f'TL文件不存在: {tl_csv_path}')

        terrain = _load_terrain_npz(terrain_txt_path)
        tl_rows = _load_tl_csv_rows(tl_csv_path, main_cfg, cfg)
        payload, meta, runtime_cfg = _build_payload(cfg, main_cfg, terrain, tl_rows)

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

        print_success('Final TL 面板数据预处理完成（Plotly版本）')
        return 0
    except Exception as exc:
        print_error(f'预处理失败: {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

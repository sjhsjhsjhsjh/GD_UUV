#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
C++ TL 三维可视化面板数据预处理脚本（Plotly版本）。

脚本功能：
1. 读取 configs/c++TL_dashboard.yaml 中的 dashboard 配置。
2. 读取 TL 文本目录中的所有 .txt 文件。
3. 解析每个文本文件：
   - 第1行：源x、源y、源z、敌x、敌y、敌z（单位：m）
   - 后续行：接收器深度、R、theta、TL（单位：m、m、deg、dB）
4. 将柱坐标（R, theta）映射到全局平面坐标（x, y）。
5. 输出前端直接可加载的 JSON：
   - payload.json
   - config_runtime.json
   - meta.json

运行示例：
    E:/lib/conda-env/torch_gpu/python.exe dashboards/c++tl_dashboard/prepare_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.interpolate import griddata

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.rich_print import print_info, print_success, print_warn  # noqa: E402


def _resolve_path(path_value: str | Path) -> Path:
    """
    将配置路径解析为项目绝对路径。

    输入参数：
        path_value: str | Path
            配置中的路径，支持相对路径和绝对路径。

    输出参数：
        Path
            解析后的绝对路径对象。

    功能说明：
        当传入相对路径时，自动以项目根目录作为基准进行拼接。

    调用示例：
        >>> _resolve_path('output/c++tl_output/')
    """
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj
    return PROJECT_ROOT / path_obj


def _load_dashboard_config(config_path: Path) -> DictConfig:
    """
    读取 C++ TL 面板配置文件。

    输入参数：
        config_path: Path
            配置文件路径，通常为 configs/c++TL_dashboard.yaml。

    输出参数：
        DictConfig
            OmegaConf 配置对象。

    功能说明：
        统一加载 dashboard 节点，供预处理与前端运行时参数生成。

    调用示例：
        >>> cfg = _load_dashboard_config(PROJECT_ROOT / 'configs' / 'c++TL_dashboard.yaml')
    """
    cfg: DictConfig = OmegaConf.load(config_path)
    return cfg


def _load_terrain(terrain_path: Path) -> dict[str, Any]:
    """
    读取地形 NPZ 并组装为前端可直接消费的数据结构。

    输入参数：
        terrain_path: Path
            地形 NPZ 路径，需包含 bathymetry_2d、x_coords、y_coords。

    输出参数：
        dict[str, Any]
            包含地形 x/y 轴（km）、深度矩阵（m）与形状信息的字典。

    功能说明：
        该函数仅做字段提取与类型转换，不改动地形坐标语义。

    调用示例：
        >>> terrain = _load_terrain(PROJECT_ROOT / 'TLdata' / 'qianshui.npz')
    """
    with np.load(terrain_path, allow_pickle=True) as terrain_npz:
        depth_m = terrain_npz['bathymetry_2d'].astype(np.float32)
        x_km = terrain_npz['x_coords'].astype(np.float32)
        y_km = terrain_npz['y_coords'].astype(np.float32)

    return {
        'x_km': x_km.tolist(),
        'y_km': y_km.tolist(),
        'depth_m': depth_m.tolist(),
        'shape': list(depth_m.shape),
        'depth_min_m': float(np.min(depth_m)),
        'depth_max_m': float(np.max(depth_m)),
    }


def _interpolate_tl_surface(x_km: np.ndarray, y_km: np.ndarray, z_m: np.ndarray, tl_db: np.ndarray, grid_size: int = 24) -> dict[str, Any]:
    """
    将散点 TL 数据插值成规则网格曲面，同时插值深度。

    输入参数：
        x_km: np.ndarray
            X 坐标（km）。
        y_km: np.ndarray
            Y 坐标（km）。
        z_m: np.ndarray
            接收器深度（m）。
        tl_db: np.ndarray
            TL 值（dB）。
        grid_size: int
            网格分辨率（每个方向的点数）。

    输出参数：
        dict[str, Any]
            包含 x_grid、y_grid、z_grid、tl_grid 的网格化曲面数据。

    功能说明：
        使用 scipy.interpolate.griddata 将散点插值到规则网格（包括深度）。

    调用示例：
        >>> surf = _interpolate_tl_surface(x_km, y_km, z_m, tl_db)
    """
    x_min, x_max = np.min(x_km), np.max(x_km)
    y_min, y_max = np.min(y_km), np.max(y_km)

    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)

    points = np.c_[x_km, y_km]
    tl_grid = griddata(points, tl_db, (xx, yy), method='cubic', fill_value=np.nanmean(tl_db))
    z_grid = griddata(points, z_m, (xx, yy), method='linear', fill_value=np.nanmean(z_m))

    return {
        'x_grid': x_grid.astype(np.float32).tolist(),
        'y_grid': y_grid.astype(np.float32).tolist(),
        'z_grid': np.nan_to_num(z_grid, nan=np.nanmean(z_m)).astype(np.float32).tolist(),
        'tl_grid': np.nan_to_num(tl_grid, nan=np.nanmean(tl_db)).astype(np.float32).tolist(),
    }


def _parse_single_txt(txt_path: Path) -> dict[str, Any]:
    """
    解析单个 TL 文本文件并转换为三维点云数据。

    输入参数：
        txt_path: Path
            单个 .txt 文件路径。

    输出参数：
        dict[str, Any]
            单文件渲染数据，包含源/敌位置、点云坐标、TL值与统计信息。

    功能说明：
        文件格式约定：
        - 第1行：source_x_m, source_y_m, source_z_m, enemy_x_m, enemy_y_m, enemy_z_m
        - 后续行：receiver_depth_m, enemy_r_m, enemy_theta_deg, tl_db

        其中后续行的 R/theta 坐标以“源点”为原点。
        映射公式：
            x_m = source_x_m + r_m * cos(theta)
            y_m = source_y_m + r_m * sin(theta)

    调用示例：
        >>> item = _parse_single_txt(PROJECT_ROOT / 'output' / 'c++tl_output' / 'sample.txt')
    """
    lines = [line.strip() for line in txt_path.read_text(encoding='utf-8').splitlines() if line.strip()]

    header = np.fromstring(lines[0], sep=' ', dtype=np.float32)
    body = np.asarray([np.fromstring(line, sep=' ', dtype=np.float32) for line in lines[1:]], dtype=np.float32)

    source_x_m = float(header[0])
    source_y_m = float(header[1])
    source_z_m = float(header[2])
    enemy_x_m = float(header[3])
    enemy_y_m = float(header[4])
    enemy_z_m = float(header[5])

    receiver_depth_m = body[:, 0].astype(np.float32)
    enemy_r_m = body[:, 1].astype(np.float32)
    enemy_theta_deg = body[:, 2].astype(np.float32)
    tl_db = body[:, 3].astype(np.float32)

    theta_rad = np.deg2rad(enemy_theta_deg)
    x_m = source_x_m + enemy_r_m * np.cos(theta_rad)
    y_m = source_y_m + enemy_r_m * np.sin(theta_rad)

    x_km = (x_m * 0.001).astype(np.float32)
    y_km = (y_m * 0.001).astype(np.float32)

    # 生成网格化曲面（同时插值深度）
    tl_surface = _interpolate_tl_surface(x_km, y_km, receiver_depth_m, tl_db, grid_size=24)

    return {
        'file_name': txt_path.name,
        'source_position_m': [source_x_m, source_y_m, source_z_m],
        'enemy_position_m': [enemy_x_m, enemy_y_m, enemy_z_m],
        'source_position_km': [source_x_m * 0.001, source_y_m * 0.001, source_z_m],
        'enemy_position_km': [enemy_x_m * 0.001, enemy_y_m * 0.001, enemy_z_m],
        'points': {
            'x_km': x_km.tolist(),
            'y_km': y_km.tolist(),
            'z_m': receiver_depth_m.tolist(),
            'r_m': enemy_r_m.tolist(),
            'theta_deg': enemy_theta_deg.tolist(),
            'tl_db': tl_db.tolist(),
        },
        'surface': tl_surface,
        'stats': {
            'point_count': int(tl_db.shape[0]),
            'tl_min_db': float(np.min(tl_db)),
            'tl_max_db': float(np.max(tl_db)),
            'tl_mean_db': float(np.mean(tl_db)),
            'depth_min_m': float(np.min(receiver_depth_m)),
            'depth_max_m': float(np.max(receiver_depth_m)),
        },
    }


def _build_payload(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    按配置读取全部 TXT 并构建 payload/meta/runtime 三类数据。

    输入参数：
        cfg: DictConfig
            预处理配置对象，需包含输入目录、输出目录、渲染参数等字段。

    输出参数：
        tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
            依次返回 payload、meta、runtime_config。

    功能说明：
        该函数是数据组织核心：
        1. 扫描并排序所有 TXT 文件；
        2. 逐文件解析为点云；
        3. 汇总全局统计；
        4. 生成前端直接使用的数据结构。

    调用示例：
        >>> payload, meta, runtime_cfg = _build_payload(cfg)
    """
    terrain_path = _resolve_path(cfg.dashboard.terrain_npz_path)
    input_dir = _resolve_path(cfg.dashboard.input_data_dir)

    terrain = _load_terrain(terrain_path)

    txt_files = sorted(input_dir.glob('*.txt'), key=lambda item: item.name)
    parsed_files = [_parse_single_txt(path) for path in txt_files]

    global_tl_all = np.concatenate([
        np.asarray(item['points']['tl_db'], dtype=np.float32)
        for item in parsed_files
    ])

    payload_files: list[dict[str, Any]] = []
    for idx, item in enumerate(parsed_files):
        payload_files.append(
            {
                'file_index': idx,
                'label': f"{idx + 1}. {item['file_name']}",
                **item,
            }
        )

    payload = {
        'metadata': {
            'file_count': int(len(payload_files)),
            'total_point_count': int(global_tl_all.shape[0]),
            'initial_file_index': int(cfg.dashboard.initial_file_index),
        },
        'terrain': {
            'x_km': terrain['x_km'],
            'y_km': terrain['y_km'],
            'depth_m': terrain['depth_m'],
            'shape': terrain['shape'],
        },
        'files': payload_files,
        'stats': {
            'tl_min_db': float(np.min(global_tl_all)),
            'tl_max_db': float(np.max(global_tl_all)),
            'terrain_depth_min_m': float(terrain['depth_min_m']),
            'terrain_depth_max_m': float(terrain['depth_max_m']),
        },
    }

    runtime_config = {
        'tl_color_min_db': None if cfg.dashboard.tl_color_min_db is None else float(cfg.dashboard.tl_color_min_db),
        'tl_color_max_db': None if cfg.dashboard.tl_color_max_db is None else float(cfg.dashboard.tl_color_max_db),
        'tl_color_cap_db': None if cfg.dashboard.tl_color_cap_db is None else float(cfg.dashboard.tl_color_cap_db),
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
        'initial_file_index': int(cfg.dashboard.initial_file_index),
    }

    meta = {
        'generated_by': 'prepare_data.py',
        'files': {
            'payload': 'payload.json',
            'runtime_config': 'config_runtime.json',
            'meta': 'meta.json',
        },
        'file_count': int(len(payload_files)),
        'total_point_count': int(global_tl_all.shape[0]),
        'terrain_shape': terrain['shape'],
        'global_tl_min_db': float(np.min(global_tl_all)),
        'global_tl_max_db': float(np.max(global_tl_all)),
    }

    if bool(cfg.dashboard.verbose_stats):
        print_info(f"文件数量: {meta['file_count']}")
        print_info(f"总点数: {meta['total_point_count']}")
        print_info(f"TL范围(dB): {meta['global_tl_min_db']:.3f} ~ {meta['global_tl_max_db']:.3f}")
        first_item = payload_files[0]
        print_warn(
            f"初始文件: {first_item['file_name']} | 点数: {first_item['stats']['point_count']} | "
            f"TL范围: {first_item['stats']['tl_min_db']:.3f} ~ {first_item['stats']['tl_max_db']:.3f}"
        )

    return payload, meta, runtime_config


def _write_json(output_dir: Path, payload: dict[str, Any], meta: dict[str, Any], runtime_cfg: dict[str, Any]) -> None:
    """
    将预处理结果写入数据目录。

    输入参数：
        output_dir: Path
            输出目录路径。
        payload: dict[str, Any]
            主数据体对象。
        meta: dict[str, Any]
            元信息对象。
        runtime_cfg: dict[str, Any]
            前端运行时配置对象。

    输出参数：
        无。

    功能说明：
        固定输出文件名为 payload.json、meta.json、config_runtime.json，
        保持与现有 dashboard 家族一致。

    调用示例：
        >>> _write_json(output_dir, payload, meta, runtime_cfg)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'payload.json').write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    (output_dir / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
    (output_dir / 'config_runtime.json').write_text(json.dumps(runtime_cfg, ensure_ascii=False), encoding='utf-8')


def main() -> int:
    """
    预处理主入口函数。

    输入参数：
        无。

    输出参数：
        int
            成功返回 0。

    功能说明：
        串联完整流程：读取配置 -> 解析输入 -> 导出 JSON。

    调用示例：
        >>> # E:/lib/conda-env/torch_gpu/python.exe dashboards/c++tl_dashboard/prepare_data.py
    """
    config_path = PROJECT_ROOT / 'configs' / 'c++TL_dashboard.yaml'
    cfg = _load_dashboard_config(config_path)
    output_dir = _resolve_path(cfg.dashboard.output_data_dir)

    print_info(f'加载配置: {config_path}')
    print_info(f"输入目录: {_resolve_path(cfg.dashboard.input_data_dir)}")
    print_info(f"地形文件: {_resolve_path(cfg.dashboard.terrain_npz_path)}")
    print_info(f'输出目录: {output_dir}')

    payload, meta, runtime_cfg = _build_payload(cfg)
    _write_json(output_dir, payload, meta, runtime_cfg)

    print_success('C++ TL 面板数据预处理完成')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

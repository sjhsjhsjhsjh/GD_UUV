#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
关键点对生成脚本（独立一键运行）
[enemy_y, uuv_x, uuv_y, uuv_z]
功能概述：
1. 读取 configs/generate_key_points.yaml 配置。
2. 读取地形文件 output/bty/terrain.npz。
3. 按配置完整枚举 enemy 与 UUV 的候选坐标。
4. 基于 terrain_3d 可通行性筛选合法点对。
5. 输出统计并保存为 npz 文件。

运行方式（项目根目录）：
    E:/lib/conda-env/torch_gpu/python.exe scripts/generate_key_points.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import yaml


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.rich_print import print_info, print_success, print_warn


def load_generate_config(config_path: Path) -> dict:
    """
    读取关键点生成配置文件。

    输入参数：
        config_path: Path
            YAML 配置文件路径。

    输出参数：
        dict
            解析后的配置字典。

    功能说明：
        读取 YAML 并返回关键点生成配置对象。

    调用示例：
        >>> cfg = load_generate_config(Path('configs/generate_key_points.yaml'))
        >>> print(cfg['file_config']['input_file'])
    """
    with config_path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def load_main_config(config_path: Path) -> dict:
    """
    读取主配置文件。

    输入参数：
        config_path: Path
            主配置文件路径。

    输出参数：
        dict
            主配置字典。

    功能说明：
        读取主配置中的地形采样间隔参数，用于坐标到网格索引的直接换算。

    调用示例：
        >>> main_cfg = load_main_config(Path('configs/main_config.yaml'))
        >>> print(main_cfg['env']['sampling_x_step'])
    """
    with config_path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def build_axis(start_value: int, end_value: int, step_value: int, axis_name: str) -> np.ndarray:
    """
    按闭区间规则构建整数坐标轴。

    输入参数：
        start_value: int
            坐标起点（包含）。
        end_value: int
            坐标终点（包含）。
        step_value: int
            坐标步长。
        axis_name: str
            当前轴名称，用于报错提示。

    输出参数：
        np.ndarray
            一维整数数组，表示离散采样坐标。

    功能说明：
        使用闭区间 [start_value, end_value] 生成坐标。

    调用示例：
        >>> axis = build_axis(1000, 1400, 100, 'uuv_x')
        >>> axis.tolist()
        [1000, 1100, 1200, 1300, 1400]
    """
    count_value = (end_value - start_value) // step_value + 1
    axis = start_value + np.arange(count_value, dtype=np.int32) * step_value
    return axis


def build_enemy_candidates(cfg: dict) -> np.ndarray:
    """
    根据配置生成 enemy 候选坐标集合。

    输入参数：
        cfg: dict
            关键点生成配置字典。

    输出参数：
        np.ndarray
            形状为 (n_enemy, 3) 的整数数组，列顺序为 [enemy_x, enemy_y, enemy_z]。

    功能说明：
        按配置约束生成 enemy 全部候选位置。当前规则为 x 和 z 固定，y 轴按步长离散采样。

    调用示例：
        >>> enemy_points = build_enemy_candidates(cfg)
        >>> enemy_points.shape[1]
        3
    """
    enemy_cfg = cfg["enemy"]
    enemy_step_cfg = cfg["enemy_step"]

    enemy_x = int(enemy_cfg["x_min"])
    enemy_z = int(enemy_cfg["z_min"])
    enemy_y_axis = build_axis(
        int(enemy_cfg["y_min"]),
        int(enemy_cfg["y_max"]),
        int(enemy_step_cfg["step_y"]),
        "enemy_y",
    )

    enemy_x_col = np.full(enemy_y_axis.shape[0], enemy_x, dtype=np.int32)
    enemy_z_col = np.full(enemy_y_axis.shape[0], enemy_z, dtype=np.int32)
    enemy_points = np.column_stack((enemy_x_col, enemy_y_axis, enemy_z_col))
    return enemy_points


def build_uuv_candidates(cfg: dict) -> np.ndarray:
    """
    根据配置生成 UUV 候选坐标集合。

    输入参数：
        cfg: dict
            关键点生成配置字典。

    输出参数：
        np.ndarray
            形状为 (n_uuv, 3) 的整数数组，列顺序为 [uuv_x, uuv_y, uuv_z]。

    功能说明：
        对 UUV 的 x/y/z 三轴按配置离散采样并做笛卡尔组合，生成完整候选点。

    调用示例：
        >>> uuv_points = build_uuv_candidates(cfg)
        >>> uuv_points[:2].tolist()
        [[1000, 1000, 50], [1000, 1000, 100]]
    """
    uuv_cfg = cfg["uuv"]
    uuv_step_cfg = cfg["UUV_step"]

    uuv_x_axis = build_axis(
        int(uuv_cfg["x_min"]),
        int(uuv_cfg["x_max"]),
        int(uuv_step_cfg["step_x"]),
        "uuv_x",
    )
    uuv_y_axis = build_axis(
        int(uuv_cfg["y_min"]),
        int(uuv_cfg["y_max"]),
        int(uuv_step_cfg["step_y"]),
        "uuv_y",
    )
    uuv_z_axis = build_axis(
        int(uuv_cfg["z_min"]),
        int(uuv_cfg["z_max"]),
        int(uuv_step_cfg["step_z"]),
        "uuv_z",
    )

    grid_x, grid_y, grid_z = np.meshgrid(uuv_x_axis, uuv_y_axis, uuv_z_axis, indexing="ij")
    uuv_points = np.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel())).astype(np.int32, copy=False)
    return uuv_points


def build_legality_mask(
    terrain_3d: np.ndarray,
    points: np.ndarray,
    sampling_x_step: int,
    sampling_y_step: int,
    sampling_z_step: int,
) -> np.ndarray:
    """
    依据 terrain_3d 计算候选点合法性掩码。

    输入参数：
        terrain_3d: np.ndarray
            三维可通行性数组，True=不可通行，False=可通行。
        points: np.ndarray
            待筛选点集，形状 (n, 3)，列顺序 [x, y, z]，单位为米。
        sampling_x_step: int
            x 轴采样间隔（单位 m）。
        sampling_y_step: int
            y 轴采样间隔（单位 m）。
        sampling_z_step: int
            z 轴采样间隔（单位 m）。

    输出参数：
        np.ndarray
            布尔掩码，True 表示该点合法（可通行）。

    功能说明：
        使用配置中的采样间隔直接将物理坐标换算为网格索引，
        并严格按 env.py 约定使用 terrain_3d[y, x, z] 进行可通行性判断。

    调用示例：
        >>> mask = build_legality_mask(terrain_3d, uuv_points, 100, 100, 50)
        >>> valid_count = int(mask.sum())
    """
    legal_mask = np.zeros(points.shape[0], dtype=bool)

    for idx_value in range(points.shape[0]):
        x_meter, y_meter, z_meter = points[idx_value]

        x_index = int(x_meter) // sampling_x_step
        y_index = int(y_meter) // sampling_y_step
        z_index = int(z_meter) // sampling_z_step - 1

        legal_mask[idx_value] = not bool(terrain_3d[y_index, x_index, z_index])

    return legal_mask


def build_key_pairs(enemy_points: np.ndarray, uuv_points: np.ndarray) -> np.ndarray:
    """
    将合法 enemy 与合法 UUV 组合为点对。

    输入参数：
        enemy_points: np.ndarray
            合法 enemy 点集，形状 (n_enemy, 3)，列顺序 [enemy_x, enemy_y, enemy_z]。
        uuv_points: np.ndarray
            合法 UUV 点集，形状 (n_uuv, 3)，列顺序 [uuv_x, uuv_y, uuv_z]。

    输出参数：
        np.ndarray
            点对数组，形状 (n_enemy*n_uuv, 4)，列顺序 [enemy_y, uuv_x, uuv_y, uuv_z]。

    功能说明：
        按固定顺序进行笛卡尔组合，形成完整初始点对定义，保证无重复、无遗漏。

    调用示例：
        >>> key_pairs = build_key_pairs(valid_enemy_points, valid_uuv_points)
        >>> key_pairs.shape[1]
        4
    """
    enemy_y = enemy_points[:, 1].astype(np.int32, copy=False)
    uuv_x = uuv_points[:, 0].astype(np.int32, copy=False)
    uuv_y = uuv_points[:, 1].astype(np.int32, copy=False)
    uuv_z = uuv_points[:, 2].astype(np.int32, copy=False)

    pair_count = enemy_y.size * uuv_x.size
    if pair_count == 0:
        return np.empty((0, 4), dtype=np.int32)

    enemy_y_col = np.repeat(enemy_y, uuv_x.size)
    uuv_x_col = np.tile(uuv_x, enemy_y.size)
    uuv_y_col = np.tile(uuv_y, enemy_y.size)
    uuv_z_col = np.tile(uuv_z, enemy_y.size)

    key_pairs = np.column_stack((enemy_y_col, uuv_x_col, uuv_y_col, uuv_z_col)).astype(np.int32, copy=False)
    return key_pairs


def save_key_points(output_file: Path, key_pairs: np.ndarray, enemy_x: int, enemy_z: int, num_output_files: int = 1) -> None:
    """
    保存合法关键点对到 npz（支持分割保存）。

    输入参数：
        output_file: Path
            输出 npz 文件路径（若分割则作为基础路径）。
        key_pairs: np.ndarray
            点对数组，列顺序 [enemy_y, uuv_x, uuv_y, uuv_z]。
        enemy_x: int
            敌方固定 x 坐标（米）。
        enemy_z: int
            敌方固定 z 坐标（米）。
        num_output_files: int
            输出文件数量，默认为1（不分割）。若>1则分割保存。

    输出参数：
        None

    功能说明：
        将筛选后的合法点对持久化到 npz。若 num_output_files=1，保存为单个文件；
        若 num_output_files>1，按行数均匀分割 key_pairs 并保存为多个编号文件。
        每个文件都包含完整的元数据（enemy_x、enemy_z、key_pair_columns）。

    调用示例：
        >>> # 单文件保存
        >>> save_key_points(Path('output/keypoints/key_points.npz'), key_pairs, 2000, 150)
        >>> # 分割保存为3个文件
        >>> save_key_points(Path('output/keypoints/key_points.npz'), key_pairs, 2000, 150, num_output_files=3)
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    key_pair_columns = np.array(["enemy_y", "uuv_x", "uuv_y", "uuv_z"])
    enemy_x_int32 = np.int32(enemy_x)
    enemy_z_int32 = np.int32(enemy_z)

    # 单文件保存（num_output_files=1）
    if num_output_files == 1:
        np.savez(
            output_file,
            key_pairs=key_pairs,
            key_pair_columns=key_pair_columns,
            enemy_x=enemy_x_int32,
            enemy_z=enemy_z_int32,
        )
    else:
        # 多文件分割保存
        key_pairs_chunks = np.array_split(key_pairs, num_output_files)
        stem = output_file.stem
        suffix = output_file.suffix
        parent_dir = output_file.parent

        for idx_value, chunk in enumerate(key_pairs_chunks):
            file_index = idx_value + 1
            indexed_filename = f"{stem}_{file_index:03d}{suffix}"
            indexed_path = parent_dir / indexed_filename

            np.savez(
                indexed_path,
                key_pairs=chunk,
                key_pair_columns=key_pair_columns,
                enemy_x=enemy_x_int32,
                enemy_z=enemy_z_int32,
            )


def main() -> int:
    """
    程序主入口（一键运行）。

    输入参数：
        无。

    输出参数：
        int
            进程退出码，0 表示成功。

    功能说明：
        串联完整流程：读配置 -> 读地形 -> 枚举候选 -> 合法性筛选 -> 组合点对 -> 保存 -> 统计输出。

    调用示例：
        >>> # 在项目根目录执行
        >>> # E:/lib/conda-env/torch_gpu/python.exe scripts/generate_key_points.py
    """
    start_time = time.perf_counter()

    config_path = project_root / "configs" / "generate_key_points.yaml"
    main_config_path = project_root / "configs" / "main_config.yaml"

    cfg = load_generate_config(config_path)
    main_cfg = load_main_config(main_config_path)

    input_file = project_root / str(cfg["file_config"]["input_file"])
    output_file = project_root / str(cfg["file_config"]["output_file"])
    num_output_files = int(cfg["file_config"].get("num_output_files", 1))

    sampling_x_step = int(main_cfg["env"]["sampling_x_step"])
    sampling_y_step = int(main_cfg["env"]["sampling_y_step"])
    sampling_z_step = int(main_cfg["env"]["sampling_z_step"])

    print_info("开始生成关键点对...")
    print_info(f"配置文件: {config_path}")
    print_info(f"输入地形: {input_file}")
    print_info(f"输出文件: {output_file}")

    terrain_data = np.load(input_file)
    terrain_3d = terrain_data["terrain_3d"]

    enemy_points = build_enemy_candidates(cfg)
    uuv_points = build_uuv_candidates(cfg)

    print_info(f"enemy 候选点数: {enemy_points.shape[0]}")
    print_info(f"uuv 候选点数: {uuv_points.shape[0]}")

    enemy_legal_mask = build_legality_mask(
        terrain_3d=terrain_3d,
        points=enemy_points,
        sampling_x_step=sampling_x_step,
        sampling_y_step=sampling_y_step,
        sampling_z_step=sampling_z_step,
    )
    uuv_legal_mask = build_legality_mask(
        terrain_3d=terrain_3d,
        points=uuv_points,
        sampling_x_step=sampling_x_step,
        sampling_y_step=sampling_y_step,
        sampling_z_step=sampling_z_step,
    )

    valid_enemy_points = enemy_points[enemy_legal_mask]
    valid_uuv_points = uuv_points[uuv_legal_mask]

    print_info(f"enemy 合法点数: {valid_enemy_points.shape[0]}")
    print_info(f"uuv 合法点数: {valid_uuv_points.shape[0]}")

    total_candidate_pairs = enemy_points.shape[0] * uuv_points.shape[0]
    key_pairs = build_key_pairs(valid_enemy_points, valid_uuv_points)
    valid_pair_count = key_pairs.shape[0]
    valid_ratio = valid_pair_count / total_candidate_pairs

    enemy_x_value = int(cfg["enemy"]["x_min"])
    enemy_z_value = int(cfg["enemy"]["z_min"])
    save_key_points(output_file, key_pairs, enemy_x_value, enemy_z_value, num_output_files)

    cost_seconds = time.perf_counter() - start_time
    print_info("=" * 64)
    print_info("关键点生成统计")
    print_info(f"候选点对总数: {total_candidate_pairs}")
    print_info(f"合法点对总数: {valid_pair_count}")
    print_info(f"合法率: {valid_ratio:.4%}")
    print_info(f"enemy 固定坐标: x={enemy_x_value}, z={enemy_z_value}")
    print_info(f"enemy 候选/合法: {enemy_points.shape[0]} / {valid_enemy_points.shape[0]}")
    print_info(f"uuv 候选/合法: {uuv_points.shape[0]} / {valid_uuv_points.shape[0]}")
    print_info(f"输出文件数量: {num_output_files}")
    if num_output_files > 1:
        rows_per_file = valid_pair_count // num_output_files
        print_info(f"每文件行数: ~{rows_per_file}（最后一个可能不同）")
    print_info(f"输出路径: {output_file}")
    print_info(f"耗时: {cost_seconds:.3f} 秒")
    print_info("=" * 64)

    if valid_pair_count == 0:
        print_warn("未筛选出任何合法点对。")
    else:
        print_success("关键点对生成完成。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""地形数据诊断工具 - 简化版本（修复索引顺序）"""

import numpy as np
import torch
from pathlib import Path
from typing import Tuple

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from utils.rich_print import print_info, print_warn, print_error, print_success


def visualize_terrain_slice(env, output_dir: Path, test_pos: Tuple[int, int, int] = (20, 20, 3)) -> None:
    """可视化地形切片并诊断坐标顺序。
    
    观测张量的形状: (1, 2, field_of_view, field_of_view, field_of_view_on_z)
    提取后: terrain_channel.shape = (field_of_view, field_of_view, field_of_view_on_z)
    
    根据 get_observation_tensor 中的赋值逻辑，索引顺序应该是 [x, y, z]
    """
    if not HAS_MATPLOTLIB:
        print_warn("matplotlib 未安装，跳过可视化")
        return

    output_dir = Path(output_dir)
    diag_dir = output_dir / "terrain_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    print_info(f"地形诊断：测试位置 {test_pos}")

    # 设置 UUV 位置
    env.reset()
    env.uuv.x, env.uuv.y, env.uuv.z = test_pos
    
    # 获取观测张量
    spatial_input, state_vector = env.get_observation_tensor(device="cuda")
    terrain_channel = spatial_input[0, 1, :, :, :].cpu().numpy()  # (field_of_view, field_of_view, field_of_view_on_z)
    
    print_info(f"观测张量形状: {spatial_input.shape}")
    print_info(f"地形通道形状: {terrain_channel.shape}")
    print_info(f"field_of_view={env.field_of_view}, field_of_view_on_z={env.field_of_view_on_z}")
    
    # 计算窗口范围
    half_fov_xy = env.field_of_view // 2
    half_fov_z = env.field_of_view_on_z // 2
    
    x_min = max(0, env.uuv.x - half_fov_xy)
    x_max = min(env.map_width - 1, env.uuv.x + half_fov_xy)
    y_min = max(0, env.uuv.y - half_fov_xy)
    y_max = min(env.map_height - 1, env.uuv.y + half_fov_xy)
    z_min = max(0, env.uuv.z - half_fov_z)
    z_max = min(env.map_depth - 1, env.uuv.z + half_fov_z)
    
    # 中心点偏移
    cx, cy, cz = env.uuv.x, env.uuv.y, env.uuv.z
    
    print_info("\n=== 关键点对比 ===")
    
    # 关键测试点
    test_points = [
        (cx, cy, cz, "中心"),
        (cx + 1, cy, cz, "中心+x"),
        (cx, cy + 1, cz, "中心+y"),
        (cx, cy, cz + 1, "中心+z"),
    ]
    
    match_xyz_count = 0  # 记录匹配 [x,y,z] 的点数
    match_xzy_count = 0  # 记录匹配 [x,z,y] 的点数
    
    for px, py, pz, label in test_points:
        if not (x_min <= px <= x_max and y_min <= py <= y_max and z_min <= pz <= z_max):
            continue
            
        raw_val = env.terrain_3d[py, px, pz]
        ox = px - x_min
        oy = py - y_min
        oz = pz - z_min
        
        print_info(f"{label} (x={px}, y={py}, z={pz}): offset=({ox}, {oy}, {oz})")
        print_info(f"  原始地形[y,x,z] = {raw_val}")
        
        # 尝试 [x,y,z] 索引
        try:
            if ox < terrain_channel.shape[0] and oy < terrain_channel.shape[1] and oz < terrain_channel.shape[2]:
                val_xyz = terrain_channel[ox, oy, oz]
                match_xyz = np.isclose(raw_val, val_xyz)
                if match_xyz:
                    match_xyz_count += 1
                print_info(f"  观测[x,y,z] = {val_xyz:.2f}, 匹配? {match_xyz}")
        except Exception as e:
            print_info(f"  观测[x,y,z] 错误: {e}")
        
        # 尝试 [x,z,y] 索引
        try:
            if ox < terrain_channel.shape[0] and oz < terrain_channel.shape[1] and oy < terrain_channel.shape[2]:
                val_xzy = terrain_channel[ox, oz, oy]
                match_xzy = np.isclose(raw_val, val_xzy)
                if match_xzy:
                    match_xzy_count += 1
                print_info(f"  观测[x,z,y] = {val_xzy:.2f}, 匹配? {match_xzy}")
        except Exception as e:
            print_info(f"  观测[x,z,y] 错误: {e}")
    
    # 诊断结论
    print_info("\n=== 诊断结论 ===")
    if match_xyz_count > match_xzy_count:
        print_success("✓ 索引顺序正确：[x, y, z]")
        correct_order = "xyz"
    elif match_xzy_count > match_xyz_count:
        print_warn("✗ 索引顺序应为：[x, z, y]")
        correct_order = "xzy"
    else:
        print_warn("⚠ 无法确定索引顺序（没有匹配的点）")
        correct_order = "unknown"
    
    # 生成图像
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 图1：原始地形 Z 切片
        ax = axes[0]
        raw_z_slice = env.terrain_3d[:, :, cz].astype(float)
        im1 = ax.imshow(raw_z_slice, cmap='gray', origin='lower')
        ax.plot(cx, cy, 'r*', markersize=15)
        ax.set_title(f'原始地形 @ z={cz}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im1, ax=ax)
        
        # 图2：观测 XY 切片（中心Z）
        ax = axes[1]
        oz_center = (cx - x_min) if correct_order == "xyz" else (cz - z_min)
        try:
            if correct_order == "xyz":
                obs_slice = terrain_channel[:, :, oz_center] if oz_center < terrain_channel.shape[2] else np.zeros((terrain_channel.shape[0], terrain_channel.shape[1]))
            elif correct_order == "xzy":
                obs_slice = terrain_channel[:, oz_center, :] if oz_center < terrain_channel.shape[1] else np.zeros((terrain_channel.shape[0], terrain_channel.shape[2]))
            else:
                obs_slice = np.zeros_like(terrain_channel[:, :, 0])
            
            im2 = ax.imshow(obs_slice.T, cmap='gray', origin='lower')
            ox_center = cx - x_min
            oy_center = cy - y_min
            ax.plot(ox_center, oy_center, 'r*', markersize=15)
            ax.set_title(f'观测张量 (顺序:{correct_order})')
            ax.set_xlabel('offset_x')
            ax.set_ylabel('offset_y')
            plt.colorbar(im2, ax=ax)
        except:
            ax.text(0.5, 0.5, 'Error in slice extraction', ha='center', va='center')
        
        plt.tight_layout()
        out_path = diag_dir / f"terrain_diagnosis_{test_pos[0]}_{test_pos[1]}_{test_pos[2]}.png"
        plt.savefig(out_path, dpi=100)
        print_success(f"图像已保存: {out_path}")
        plt.close()
    except Exception as e:
        print_warn(f"生成图像失败: {e}")
    
    # 生成报告
    try:
        report_path = diag_dir / f"terrain_diagnosis_report_{test_pos[0]}_{test_pos[1]}_{test_pos[2]}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("地形诊断报告\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"测试位置: {test_pos}\n")
            f.write(f"地形形状: {env.terrain_3d.shape}\n")
            f.write(f"观测张量形状: {spatial_input.shape}\n")
            f.write(f"地形通道形状: {terrain_channel.shape}\n")
            f.write(f"field_of_view: {env.field_of_view}\n")
            f.write(f"field_of_view_on_z: {env.field_of_view_on_z}\n\n")
            f.write(f"诊断结果: 索引顺序为 [{correct_order}]\n")
            f.write(f"匹配 [x,y,z]: {match_xyz_count}/4\n")
            f.write(f"匹配 [x,z,y]: {match_xzy_count}/4\n")
        print_success(f"报告已保存: {report_path}")
    except Exception as e:
        print_warn(f"生成报告失败: {e}")
    
    print_info("诊断完成")

"""地形数据诊断工具 - 修复版本 v3（使用GPU张量比较）"""

import numpy as np
import torch
from pathlib import Path
from typing import Tuple

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from utils.rich_print import print_info, print_warn, print_success


def visualize_terrain_slice(env, output_dir: Path, test_pos: Tuple[int, int, int] = (20, 20, 3)) -> None:
    """可视化地形切片并诊断坐标顺序。
    
    观测张量的形状: (1, 2, field_of_view, field_of_view, field_of_view_on_z)
    提取后: terrain_channel.shape = (field_of_view, field_of_view, field_of_view_on_z)
    
    索引顺序: get_observation_tensor 使用 terrain_3d_tensor[x, y, z] 索引
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
    
    # 中心点
    cx, cy, cz = env.uuv.x, env.uuv.y, env.uuv.z
    
    print_info("\n=== 关键点对比 ===")
    
    # 关键测试点
    test_points = [
        (cx, cy, cz, "中心"),
        (cx + 1, cy, cz, "中心+x" if cx + 1 <= x_max else "中心+x(越界)"),
        (cx, cy + 1, cz, "中心+y" if cy + 1 <= y_max else "中心+y(越界)"),
        (cx, cy, cz + 1, "中心+z" if cz + 1 <= z_max else "中心+z(越界)"),
    ]
    
    match_xyz_count = 0  # 记录匹配的点数
    
    for px, py, pz, label in test_points:
        if not (x_min <= px <= x_max and y_min <= py <= y_max and z_min <= pz <= z_max):
            print_info(f"{label} (x={px}, y={py}, z={pz}): 在窗口外，跳过")
            continue
        
        # 从GPU张量中获取原始值
        raw_val = env.terrain_3d_tensor[px, py, pz].item()
        
        # 计算观测张量中的偏移
        ox = px - x_min
        oy = py - y_min
        oz = pz - z_min
        
        print_info(f"{label} (x={px}, y={py}, z={pz}): offset=({ox}, {oy}, {oz})")
        print_info(f"  原始张量[x,y,z] = {raw_val:.2f}")
        
        # 检查索引是否在观测张量范围内
        if ox < terrain_channel.shape[0] and oy < terrain_channel.shape[1] and oz < terrain_channel.shape[2]:
            obs_val = terrain_channel[ox, oy, oz]
            match = np.isclose(raw_val, obs_val)
            if match:
                match_xyz_count += 1
            status = "[OK]" if match else "[FAIL]"
            print_info(f"  观测张量[ox,oy,oz] = {obs_val:.2f}, 匹配? {match} {status}")
        else:
            print_warn(f"  偏移 ({ox}, {oy}, {oz}) 超出观测张量范围 {terrain_channel.shape}")
    
    # 诊断结论
    print_info("\n=== 诊断结论 ===")
    if match_xyz_count >= 1:
        print_success(f"[OK] 索引顺序正确：[x, y, z] (匹配 {match_xyz_count}/3 个有效点)")
    else:
        print_warn(f"[FAIL] 索引顺序可能有问题 (匹配 {match_xyz_count}/3 个有效点)")
    
    # 生成图像
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 图1：原始地形 Z 切片（使用GPU张量）
        ax = axes[0]
        raw_z_slice = env.terrain_3d_tensor[:, :, cz].cpu().numpy()
        im1 = ax.imshow(raw_z_slice.T, cmap='gray', origin='lower')  # 转置以匹配 [x,y] 显示
        ax.plot(cx, cy, 'r*', markersize=15, label='UUV')
        ax.set_title(f'原始地形张量 @ z={cz}\n(索引: [x,y,z])')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        plt.colorbar(im1, ax=ax)
        
        # 图2：观测 XY 切片（中心Z）
        ax = axes[1]
        oz_center = cz - z_min
        if 0 <= oz_center < terrain_channel.shape[2]:
            obs_slice = terrain_channel[:, :, oz_center]
            im2 = ax.imshow(obs_slice.T, cmap='gray', origin='lower')  # 转置以匹配显示
            ox_center = cx - x_min
            oy_center = cy - y_min
            ax.plot(ox_center, oy_center, 'r*', markersize=15, label='观测中心')
            ax.set_title(f'观测张量通道1 @ offset_z={oz_center}\n(索引: [ox,oy,oz])')
            ax.set_xlabel('offset_x')
            ax.set_ylabel('offset_y')
            ax.legend()
            plt.colorbar(im2, ax=ax)
        else:
            ax.text(0.5, 0.5, f'切片超出范围 (oz={oz_center})', ha='center', va='center')
        
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
            f.write(f"地形张量形状: {env.terrain_3d_tensor.shape}\n")
            f.write(f"观测张量形状: {spatial_input.shape}\n")
            f.write(f"地形通道形状: {terrain_channel.shape}\n")
            f.write(f"field_of_view: {env.field_of_view}\n")
            f.write(f"field_of_view_on_z: {env.field_of_view_on_z}\n\n")
            f.write(f"诊断结论: 索引顺序为 [x, y, z]\n")
            f.write(f"关键点对比结果: 匹配 {match_xyz_count} 个有效点\n")
        print_success(f"报告已保存: {report_path}")
    except Exception as e:
        print_warn(f"生成报告失败: {e}")
    
    print_info("诊断完成")

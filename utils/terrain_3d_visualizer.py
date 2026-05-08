"""三维地形可视化工具 - 直观展示坐标系"""

import numpy as np
import torch
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.rich_print import print_info, print_success, print_error


def visualize_terrain_3d(env, output_dir: Path, test_pos: Tuple[int, int, int] = (20, 20, 3)) -> None:
    """生成三维地形可视化，展示 UUV 位置和观测窗口。
    
    这个脚本直观展示地形坐标系统，让用户一眼看出索引顺序是否正确。
    """
    
    output_dir = Path(output_dir)
    viz_dir = output_dir / "terrain_3d_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    print_info(f"生成三维地形可视化，UUV 位置: {test_pos}")

    # 设置 UUV 位置
    env.reset()
    env.uuv.x, env.uuv.y, env.uuv.z = test_pos
    
    # 获取观测张量
    spatial_input, state_vector = env.get_observation_tensor(device="cuda")
    terrain_channel = spatial_input[0, 1, :, :, :].cpu().numpy()
    
    print_info(f"UUV 位置: ({env.uuv.x}, {env.uuv.y}, {env.uuv.z})")
    print_info(f"地形张量形状: {env.terrain_3d_tensor.shape}")
    print_info(f"观测张量形状: {spatial_input.shape}")

    # ============ 方案 A: 在实际地形中绘制关键点 ============
    fig = plt.figure(figsize=(16, 6))

    # 绘制 1: 原始地形中的关键点
    ax1 = fig.add_subplot(131, projection='3d')
    
    # 提取所有 terrain_3d_tensor 中值为 1.0 的点（可通行区域）
    walkable_points = []
    walkable_indices = []
    for x in range(0, env.map_width, 5):  # 间隔采样
        for y in range(0, env.map_height, 5):
            for z in range(0, env.map_depth):
                val = env.terrain_3d_tensor[x, y, z].item()
                if val > 0.5:  # 可通行
                    walkable_points.append([x, y, z])
                    walkable_indices.append((x, y, z))
    
    if walkable_points:
        walkable_points = np.array(walkable_points)
        ax1.scatter(walkable_points[:, 0], walkable_points[:, 1], walkable_points[:, 2], 
                   c='blue', s=10, alpha=0.3, label='可通行')
    
    # 标记 UUV 位置
    ax1.scatter([env.uuv.x], [env.uuv.y], [env.uuv.z], 
               c='red', s=200, marker='*', label='UUV')
    
    # 标记观测窗口的对角线
    half_fov_xy = env.field_of_view // 2
    half_fov_z = env.field_of_view_on_z // 2
    
    x_min = max(0, env.uuv.x - half_fov_xy)
    x_max = min(env.map_width - 1, env.uuv.x + half_fov_xy)
    y_min = max(0, env.uuv.y - half_fov_xy)
    y_max = min(env.map_height - 1, env.uuv.y + half_fov_xy)
    z_min = max(0, env.uuv.z - half_fov_z)
    z_max = min(env.map_depth - 1, env.uuv.z + half_fov_z)
    
    # 绘制观测窗口的边界框
    box_corners = [
        [x_min, y_min, z_min], [x_max, y_min, z_min],
        [x_min, y_max, z_min], [x_max, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max],
        [x_min, y_max, z_max], [x_max, y_max, z_max],
    ]
    box_corners = np.array(box_corners)
    ax1.scatter(box_corners[:, 0], box_corners[:, 1], box_corners[:, 2], 
               c='green', s=100, marker='^', label='观测窗口角')
    
    ax1.set_xlabel('X (Width)')
    ax1.set_ylabel('Y (Height)')
    ax1.set_zlabel('Z (Depth)')
    ax1.set_title(f'原始地形 - UUV @ ({env.uuv.x}, {env.uuv.y}, {env.uuv.z})')
    ax1.legend()
    ax1.set_xlim(0, env.map_width)
    ax1.set_ylim(0, env.map_height)
    ax1.set_zlim(0, env.map_depth)

    # ============ 方案 B: 观测张量中的关键点 ============
    ax2 = fig.add_subplot(132, projection='3d')
    
    obs_walkable = []
    for ox in range(terrain_channel.shape[0]):
        for oy in range(terrain_channel.shape[1]):
            for oz in range(terrain_channel.shape[2]):
                if terrain_channel[ox, oy, oz] > 0.5:
                    obs_walkable.append([ox, oy, oz])
    
    if obs_walkable:
        obs_walkable = np.array(obs_walkable)
        ax2.scatter(obs_walkable[:, 0], obs_walkable[:, 1], obs_walkable[:, 2],
                   c='blue', s=10, alpha=0.3, label='观测中可通行')
    
    # 标记观测中心
    offset_x = env.uuv.x - x_min
    offset_y = env.uuv.y - y_min
    offset_z = env.uuv.z - z_min
    
    ax2.scatter([offset_x], [offset_y], [offset_z],
               c='red', s=200, marker='*', label='观测中心')
    
    ax2.set_xlabel('offset_x')
    ax2.set_ylabel('offset_y')
    ax2.set_zlabel('offset_z')
    ax2.set_title(f'观测张量 - 中心 @ ({offset_x}, {offset_y}, {offset_z})')
    ax2.legend()
    ax2.set_xlim(0, terrain_channel.shape[0])
    ax2.set_ylim(0, terrain_channel.shape[1])
    ax2.set_zlim(0, terrain_channel.shape[2])

    # ============ 方案 C: 关键点对比 ============
    ax3 = fig.add_subplot(133)
    
    # 测试几个关键点
    test_points = [
        (env.uuv.x, env.uuv.y, env.uuv.z, "中心"),
        (env.uuv.x + 1, env.uuv.y, env.uuv.z, "中心+x"),
        (env.uuv.x, env.uuv.y + 1, env.uuv.z, "中心+y"),
        (env.uuv.x, env.uuv.y, env.uuv.z + 1, "中心+z"),
    ]
    
    comparison_text = "关键点对比 (索引顺序检验):\n\n"
    match_count = 0
    
    for px, py, pz, label in test_points:
        if not (x_min <= px <= x_max and y_min <= py <= y_max and z_min <= pz <= z_max):
            comparison_text += f"{label}: 在窗口外\n"
            continue
        
        # 原始地形张量值 - 使用 [x, y, z] 索引
        raw_xyz = env.terrain_3d_tensor[px, py, pz].item()
        
        # 观测张量值 - 使用相同索引顺序
        ox = px - x_min
        oy = py - y_min
        oz = pz - z_min
        obs_xyz = terrain_channel[ox, oy, oz]
        
        match = np.isclose(raw_xyz, obs_xyz)
        match_count += match
        status = "[MATCH]" if match else "[MISMATCH]"
        comparison_text += f"{label}: raw={raw_xyz:.1f}, obs={obs_xyz:.1f} {status}\n"
    
    comparison_text += f"\n匹配度: {match_count}/3\n"
    
    if match_count >= 2:
        comparison_text += "\n结论: 索引顺序 [x, y, z] 正确!"
        ax3.text(0.1, 0.5, comparison_text, fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    else:
        comparison_text += "\n结论: 索引顺序有问题，应该是 [y, x, z]"
        ax3.text(0.1, 0.5, comparison_text, fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    ax3.axis('off')

    plt.tight_layout()
    out_path = viz_dir / f"terrain_3d_visualization_{test_pos[0]}_{test_pos[1]}_{test_pos[2]}.png"
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    print_success(f"3D 可视化已保存: {out_path}")
    plt.close()

    # ============ 生成额外的 2D 切片对比 ============
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 三个不同的 Z 切片
    for z_idx, z_val in enumerate([z_min, env.uuv.z, z_max]):
        if z_val >= env.map_depth:
            continue
        
        # 原始地形
        ax = axes[0, z_idx]
        raw_slice = env.terrain_3d_tensor[:, :, z_val].cpu().numpy()
        im = ax.imshow(raw_slice.T, cmap='gray', origin='lower')
        ax.scatter([env.uuv.x], [env.uuv.y], c='red', s=100, marker='*')
        ax.set_title(f'原始地形 Z={z_val}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax)
        
        # 观测张量
        ax = axes[1, z_idx]
        oz = z_val - z_min
        if 0 <= oz < terrain_channel.shape[2]:
            obs_slice = terrain_channel[:, :, oz]
            im = ax.imshow(obs_slice.T, cmap='gray', origin='lower')
            ax.scatter([offset_x], [offset_y], c='red', s=100, marker='*')
            ax.set_title(f'观测张量 offset_z={oz}')
            ax.set_xlabel('offset_x')
            ax.set_ylabel('offset_y')
            plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    slice_path = viz_dir / f"terrain_2d_slices_{test_pos[0]}_{test_pos[1]}_{test_pos[2]}.png"
    plt.savefig(slice_path, dpi=100, bbox_inches='tight')
    print_success(f"2D 切片已保存: {slice_path}")
    plt.close()

    print_info("可视化完成！请查看生成的 PNG 文件来判断坐标系。")

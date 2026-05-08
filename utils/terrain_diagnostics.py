"""地形数据诊断工具：验证坐标顺序和观测张量正确性。

功能：
1. 可视化原始地形数据
2. 可视化网络观测张量
3. 对比验证坐标对应关系
4. 生成诊断报告和图像
"""

import numpy as np
import torch
from pathlib import Path
from typing import Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from utils.rich_print import print_info, print_warn, print_error, print_success


def visualize_terrain_slice(env, output_dir: Path, test_pos: Tuple[int, int, int] = (20, 20, 3)) -> None:
    """可视化地形切片，验证坐标顺序的正确性。

    功能说明:
        1. 将 UUV 设置到指定位置
        2. 调用 get_observation_tensor 获取观测张量
        3. 对比原始地形数据和观测张量
        4. 生成可视化图像并保存到 output 目录

    输入参数:
        env: 环境对象
        output_dir: 输出目录路径
        test_pos: 测试位置 (x, y, z)，默认 (20, 20, 3)

    输出参数:
        无。生成的图像保存到 output_dir/terrain_diagnostics/
    """
    if not HAS_MATPLOTLIB:
        print_warn("matplotlib 未安装，跳过地形诊断可视化")
        return

    output_dir = Path(output_dir)
    diag_dir = output_dir / "terrain_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    print_info(f"开始地形数据诊断，测试位置: {test_pos}")

    # === 步骤1：设置 UUV 位置 ===
    env.reset()
    env.uuv.x, env.uuv.y, env.uuv.z = test_pos
    print_info(f"设置 UUV 位置: ({env.uuv.x}, {env.uuv.y}, {env.uuv.z})")

    # === 步骤2：获取观测张量 ===
    spatial_input, state_vector = env.get_observation_tensor(device="cuda")
    print_info(f"观测张量形状: spatial_input={spatial_input.shape}, state_vector={state_vector.shape}")

    # 从 GPU 转移到 CPU 用于可视化
    terrain_channel = spatial_input[0, 1, :, :, :].cpu().numpy()  # 通道 1: 地形
    tl_channel = spatial_input[0, 0, :, :, :].cpu().numpy()       # 通道 0: TL

    print_info(f"地形通道形状: {terrain_channel.shape}")
    print_info(f"TL 通道形状: {tl_channel.shape}")

    # === 步骤3：对比原始地形 ===
    print_info("\n=== 坐标对应验证 ===")
    
    # 获取观测窗口的范围
    half_fov_xy = env.field_of_view // 2
    half_fov_z = env.field_of_view_on_z // 2
    
    x_min = max(0, env.uuv.x - half_fov_xy)
    x_max = min(env.map_width - 1, env.uuv.x + half_fov_xy)
    y_min = max(0, env.uuv.y - half_fov_xy)
    y_max = min(env.map_height - 1, env.uuv.y + half_fov_xy)
    z_min = max(0, env.uuv.z - half_fov_z)
    z_max = min(env.map_depth - 1, env.uuv.z + half_fov_z)

    print_info(f"观测窗口（地图坐标）: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}], z=[{z_min}, {z_max}]")

    # 提取原始地形数据（注意原始地形的索引顺序）
    print_info("\n原始地形访问方式验证:")
    print_info(f"  terrain_3d 原始形状: {env.terrain_3d.shape}")
    print_info(f"  terrain_3d[y, x, z] 访问方式（期望）")

    # 取中心点附近的一个小区域进行对比
    center_x, center_y, center_z = env.uuv.x, env.uuv.y, env.uuv.z
    offset_x = center_x - x_min
    offset_y = center_y - y_min
    offset_z = center_z - z_min

    print_info(f"\n中心点在观测张量中的偏移: offset_x={offset_x}, offset_y={offset_y}, offset_z={offset_z}")

    # 对比几个关键点
    print_info("\n关键点对比（原始地形 vs 观测张量）:")
    test_points = [
        (center_x, center_y, center_z, "中心"),
        (center_x + 1, center_y, center_z, "中心+x"),
        (center_x, center_y + 1, center_z, "中心+y"),
        (center_x, center_y, center_z + 1, "中心+z"),
    ]

    for px, py, pz, label in test_points:
        if x_min <= px <= x_max and y_min <= py <= y_max and z_min <= pz <= z_max:
            # 原始地形值
            raw_terrain = env.terrain_3d[py, px, pz]
            
            # 观测张量中的值（需要转换坐标）
            ox = px - x_min
            oy = py - y_min
            oz = pz - z_min
            
            # 尝试两种可能的索引方式
            obs_val_xyz = terrain_channel[oz, oy, ox]
            obs_val_yxz = terrain_channel[oz, ox, oy]
            
            print_info(f"  {label} (x={px}, y={py}, z={pz}):")
            print_info(f"    原始地形[y,x,z]: {raw_terrain}")
            print_info(f"    观测张量[z,y,x]: {obs_val_xyz:.2f}")
            print_info(f"    观测张量[z,x,y]: {obs_val_yxz:.2f}")
            print_info(f"    匹配 [z,y,x]? {np.isclose(raw_terrain, obs_val_xyz)}")
            print_info(f"    匹配 [z,x,y]? {np.isclose(raw_terrain, obs_val_yxz)}")

    # === 步骤4：生成可视化 ===
    print_info("\n生成可视化图像...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"地形诊断：UUV @ ({env.uuv.x}, {env.uuv.y}, {env.uuv.z})", fontsize=16)

    # 图 1: 原始地形的 Z=center_z 切片
    ax = axes[0, 0]
    raw_slice_z = env.terrain_3d[:, :, center_z].astype(float)
    im1 = ax.imshow(raw_slice_z, cmap='gray', origin='lower')
    ax.plot(center_x, center_y, 'r*', markersize=15, label='UUV')
    ax.set_title(f'原始地形 [y, x] @ z={center_z}\n(索引顺序: terrain_3d[y, x, z])')
    ax.set_xlabel('x 坐标')
    ax.set_ylabel('y 坐标')
    ax.legend()
    plt.colorbar(im1, ax=ax)

    # 图 2: 观测张量的 Z=offset_z 切片（假设索引是 [z, y, x]）
    ax = axes[0, 1]
    if offset_z < terrain_channel.shape[0]:
        obs_slice_z = terrain_channel[offset_z, :, :]
        im2 = ax.imshow(obs_slice_z, cmap='gray', origin='lower')
        ax.plot(offset_x, offset_y, 'r*', markersize=15, label='观测中心')
        ax.set_title(f'观测张量通道1 [offset_y, offset_x] @ offset_z={offset_z}\n(假设索引: [z, y, x])')
        ax.set_xlabel('offset_x')
        ax.set_ylabel('offset_y')
        ax.legend()
        plt.colorbar(im2, ax=ax)

    # 图 3: 原始地形的 Y=center_y 切片
    ax = axes[1, 0]
    raw_slice_y = env.terrain_3d[center_y, :, :].astype(float)
    im3 = ax.imshow(raw_slice_y, cmap='gray', origin='lower')
    ax.plot(center_x, center_z, 'r*', markersize=15, label='UUV')
    ax.set_title(f'原始地形 [x, z] @ y={center_y}\n(索引顺序: terrain_3d[y, x, z])')
    ax.set_xlabel('x 坐标')
    ax.set_ylabel('z 坐标')
    ax.legend()
    plt.colorbar(im3, ax=ax)

    # 图 4: TL 通道的 Z 切片
    ax = axes[1, 1]
    if offset_z < tl_channel.shape[0]:
        tl_slice_z = tl_channel[offset_z, :, :]
        im4 = ax.imshow(tl_slice_z, cmap='hot', origin='lower')
        ax.plot(offset_x, offset_y, 'b*', markersize=15, label='观测中心')
        ax.set_title(f'TL 通道 [offset_y, offset_x] @ offset_z={offset_z}')
        ax.set_xlabel('offset_x')
        ax.set_ylabel('offset_y')
        ax.legend()
        plt.colorbar(im4, ax=ax)

    # 保存图像
    output_path = diag_dir / f"terrain_diagnosis_pos_{env.uuv.x}_{env.uuv.y}_{env.uuv.z}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print_success(f"诊断图像已保存: {output_path}")

    plt.close()

    # === 步骤5：生成文本报告 ===
    report_path = diag_dir / f"terrain_diagnosis_report_pos_{env.uuv.x}_{env.uuv.y}_{env.uuv.z}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("地形数据诊断报告\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"测试位置: ({env.uuv.x}, {env.uuv.y}, {env.uuv.z})\n")
        f.write(f"地图尺寸: {env.map_size}\n")
        f.write(f"field_of_view: {env.field_of_view}\n")
        f.write(f"field_of_view_on_z: {env.field_of_view_on_z}\n\n")

        f.write("原始地形信息:\n")
        f.write(f"  - 形状: {env.terrain_3d.shape}\n")
        f.write(f"  - dtype: {env.terrain_3d.dtype}\n")
        f.write(f"  - 索引方式: terrain_3d[y, x, z]\n")
        f.write(f"  - 中心点值: terrain_3d[{center_y}, {center_x}, {center_z}] = {env.terrain_3d[center_y, center_x, center_z]}\n\n")

        f.write("观测张量信息:\n")
        f.write(f"  - 形状: {spatial_input.shape}\n")
        f.write(f"  - dtype: {spatial_input.dtype}\n")
        f.write(f"  - 通道0 (TL): {tl_channel.shape}\n")
        f.write(f"  - 通道1 (地形): {terrain_channel.shape}\n\n")

        f.write("坐标对应关系验证:\n")
        for px, py, pz, label in test_points:
            if x_min <= px <= x_max and y_min <= py <= y_max and z_min <= pz <= z_max:
                raw_terrain = env.terrain_3d[py, px, pz]
                ox, oy, oz = px - x_min, py - y_min, pz - z_min
                obs_val_xyz = terrain_channel[oz, oy, ox]
                obs_val_yxz = terrain_channel[oz, ox, oy]
                f.write(f"  {label} (x={px}, y={py}, z={pz}):\n")
                f.write(f"    原始地形[y,x,z] = {raw_terrain}\n")
                f.write(f"    观测张量[z,y,x] = {obs_val_xyz:.2f} (匹配? {np.isclose(raw_terrain, obs_val_xyz)})\n")
                f.write(f"    观测张量[z,x,y] = {obs_val_yxz:.2f} (匹配? {np.isclose(raw_terrain, obs_val_yxz)})\n\n")

    print_success(f"诊断报告已保存: {report_path}")
    print_info("地形诊断完成")

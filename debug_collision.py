#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from env.env import Env
from utils.rich_print import print_info, print_warn, print_error
from omegaconf import OmegaConf

# 加载配置
cfg = OmegaConf.load('configs/main_config.yaml')

# 初始化环境
env = Env(cfg)

print_info("=" * 60)
print_info("撞山测试 - 调试版本")
print_info("=" * 60)

# 重置环境
env.reset()
print_info(f"环境重置，UUV 初始位置: ({env.uuv.x}, {env.uuv.y}, {env.uuv.z})")

# 在有效范围内寻找第一个山体
mountain_found = False
print_info(f"UUV 有效范围: x=[{env.uuv_x_min_index}, {env.uuv_x_max_index}], "
           f"y=[{env.uuv_y_min_index}, {env.uuv_y_max_index}], "
           f"z=[{env.uuv_z_min_index}, {env.uuv_z_max_index}]")
print_info(f"在有效范围内寻找山体...")

for y in range(env.uuv_y_min_index, env.uuv_y_max_index + 1):
    for x in range(env.uuv_x_min_index, env.uuv_x_max_index + 1):
        for z in range(env.uuv_z_min_index, env.uuv_z_max_index + 1):
            if env.terrain_3d[y, x, z] == True:
                mnt_x, mnt_y, mnt_z = x, y, z
                print_info(f"找到山体: ({mnt_x}, {mnt_y}, {mnt_z})")
                mountain_found = True
                break
        if mountain_found:
            break
    if mountain_found:
        break

if mountain_found:
    # 寻找邻近的可通行点
    print_info(f"寻找山体 ({mnt_x}, {mnt_y}, {mnt_z}) 的邻近可通行点...")
    found_nearby = False
    for nearby_z in [mnt_z - 1, mnt_z + 1]:
        print_info(f"  检查 z={nearby_z}...")
        if 0 <= nearby_z < env.map_depth:
            is_passable = not env.terrain_3d[mnt_y, mnt_x, nearby_z]
            print_info(f"    可通行? {is_passable}")
            if is_passable:
                # 设置 UUV 位置
                env.uuv.x = mnt_x
                env.uuv.y = mnt_y
                env.uuv.z = nearby_z
                print_info(f"    设置 UUV 位置为 ({env.uuv.x}, {env.uuv.y}, {env.uuv.z})")
                
                # 验证位置设置成功
                if env.uuv.x == mnt_x and env.uuv.y == mnt_y and env.uuv.z == nearby_z:
                    print_info(f"    位置验证成功")
                    found_nearby = True
                else:
                    print_warn(f"    位置验证失败: ({env.uuv.x}, {env.uuv.y}, {env.uuv.z})")
                break
    
    if found_nearby:
        print_info(f"找到邻近可通行点，准备执行撞山动作...")
        print_info(f"UUV 当前位置: ({env.uuv.x}, {env.uuv.y}, {env.uuv.z})")
        print_info(f"山体位置: ({mnt_x}, {mnt_y}, {mnt_z})")
        
        # 根据方向执行动作
        if mnt_z > env.uuv.z:
            print_info(f"执行动作 5 (向上)，期望从 z={env.uuv.z} 移动到 z={env.uuv.z + 1}")
            action = 5
        else:
            print_info(f"执行动作 4 (向下)，期望从 z={env.uuv.z} 移动到 z={env.uuv.z - 1}")
            action = 4
        
        state, reward, done, info = env.step(action)
        
        print_info(f"执行后 - UUV 位置: ({env.uuv.x}, {env.uuv.y}, {env.uuv.z})")
        print_info(f"done={done}, reward={reward}, result={info.get('result', 'N/A')}")
        
        if done and reward == -100 and "撞山" in info.get('result', ''):
            print_info(f"✓ 撞山条件正确触发！")
        else:
            print_warn(f"✗ 撞山条件未触发")
            print_warn(f"  期望: done=True, reward=-100, result 包含 '撞山'")
            print_warn(f"  实际: done={done}, reward={reward}, result={info.get('result', 'N/A')}")
    else:
        print_warn("未找到邻近的可通行点")
else:
    print_warn("未找到山体")

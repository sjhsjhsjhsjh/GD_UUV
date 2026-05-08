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
print_info("被发现条件测试 - 调试版本")
print_info("=" * 60)

# 重置环境
env.reset()
print_info(f"环境重置，UUV 初始位置: ({env.uuv.x}, {env.uuv.y}, {env.uuv.z})")
print_info(f"敌方位置: ({env.enemy.x}, {env.enemy.y}, {env.enemy.z})")
print_info(f"TL 参数: tl_proper={env.tl_proper}, tl_tolerance={env.tl_tolerance}")
print_info(f"被发现阈值: {env.tl_tolerance * 2.2}")

# 多步靠近敌人
detected = False
max_steps = 500
for step_idx in range(max_steps):
    state, reward, done, info = env.step(0)  # 不断靠近（动作 0）
    
    if step_idx == 0 or step_idx % 50 == 0 or done:
        print_info(f"Step {step_idx}: UUV=({env.uuv.x}, {env.uuv.y}, {env.uuv.z}), "
                  f"acoustic_signal={env.cumulative_acoustic_signal:.2f}, done={done}, "
                  f"reward={reward:.4f}, result={info.get('result', 'N/A')}")
    
    if done:
        print_info(f"\n✓ 在第 {step_idx + 1} 步触发结束条件")
        print_info(f"  结果: {info.get('result', 'N/A')}")
        print_info(f"  奖励: {reward}")
        print_info(f"  最终 UUV 位置: ({env.uuv.x}, {env.uuv.y}, {env.uuv.z})")
        print_info(f"  最终累计声信号强度: {env.cumulative_acoustic_signal:.2f}")
        
        if "被发现" in info.get('result', ''):
            detected = True
        break

if not detected:
    print_warn(f"✗ 执行 {max_steps} 步后未触发被发现条件")
    print_warn(f"  最终 UUV 位置: ({env.uuv.x}, {env.uuv.y}, {env.uuv.z})")
    print_warn(f"  最终累计声信号强度: {env.cumulative_acoustic_signal:.2f}")

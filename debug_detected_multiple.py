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
print_info("被发现条件多次试验 - 测试成功率")
print_info("=" * 60)

detected_count = 0
total_trials = 10

for trial in range(total_trials):
    env.reset()
    detected_in_trial = False
    
    for step_idx in range(1000):  # 改为 1000 步
        state, reward, done, info = env.step(0)
        
        if done and reward == -10 and "被发现" in info.get('result', ''):
            detected_in_trial = True
            detected_count += 1
            break
    
    status = "✓" if detected_in_trial else "✗"
    print_info(f"试验 {trial + 1}: {status}")

print_info(f"\n成功率: {detected_count}/{total_trials} ({detected_count*100//total_trials}%)")

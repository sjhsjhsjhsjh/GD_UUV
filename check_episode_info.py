#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查日志中的 enemy_y 值
"""

import json
from pathlib import Path

log_file = Path('outputs/2026-05-14/22-26-45/step_log.txt')
print(f'检查日志文件: {log_file}')

if log_file.exists():
    with open(log_file) as f:
        for i, line in enumerate(f):
            if 'EPISODE_INFO' in line:
                try:
                    part = line.split('EPISODE_INFO:', 1)[1].strip()
                    data = json.loads(part)
                    print(f'行 {i}: enemy_y={data.get("enemy_y")}, '
                          f'enemy_direction={data.get("enemy_direction")}')
                    break
                except Exception as e:
                    print(f'解析第 {i} 行失败: {e}')

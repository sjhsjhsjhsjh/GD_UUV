#!/usr/bin/env python3
import requests
import json

# 测试 replay-episode API
log_path = r'E:\program\GD-UUV-self\outputs\2026-05-13\19-08-10\step_log.txt'
episode_index = 10

print("=" * 70)
print("测试 replay-episode API")
print("=" * 70)

try:
    response = requests.post(
        'http://127.0.0.1:5000/api/replay-episode',
        json={'log_path': log_path, 'episode_index': episode_index},
        timeout=60
    )
    
    if response.status_code == 200:
        data = response.json()
        trajectory = data['trajectory']
        
        print(f"\n✓ 成功回放 Episode #{trajectory['episode_index']}")
        print(f"  - 总步数: {len(trajectory['trajectory'])}")
        print(f"  - 初始位置: UUV({trajectory['episode_info']['uuv_x']}, {trajectory['episode_info']['uuv_y']}, {trajectory['episode_info']['uuv_z']})")
        print(f"  - 敌方初始Y: {trajectory['episode_info']['enemy_y']}")
        print(f"\n第一步详情:")
        first_step = trajectory['trajectory'][0]
        print(json.dumps(first_step, indent=2, ensure_ascii=False))
        
        print(f"\n最后一步详情:")
        last_step = trajectory['trajectory'][-1]
        print(json.dumps(last_step, indent=2, ensure_ascii=False))
        
    else:
        print(f"✗ API 返回错误: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("=" * 70)

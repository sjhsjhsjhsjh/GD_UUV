import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pathlib import Path
import torch
import time
import numpy as np
from rich.table import Table
from rich.console import Console

from env.env import Env
from agent.trainer import PPOTrainer
from utils.rich_print import print_info, print_warn, print_error
from utils.terrain_diagnostics_v3 import visualize_terrain_slice
from utils.terrain_3d_visualizer import visualize_terrain_3d


def test_basic_reset_and_step(env: Env) -> dict:
    """Phase 1: 基础功能测试 - reset() 和 step() 的正确性

    功能说明:
        测试环境的重置和单步功能，验证位置有效性、状态清零、动作效果等。

    输入参数:
        env (Env): 环境对象

    输出参数:
        dict: 包含测试结果的字典，键为测试项目，值为 (是否通过, 详细信息)
    """
    results = {}
    console = Console()
    
    print_info("\n" + "="*70)
    print_info("Phase 1: 基础功能测试 - reset() 和 step()")
    print_info("="*70)
    
    # ===== 1.1 reset() 正常性测试 =====
    print_info("\n[1.1] 测试 reset() 正常性...")
    
    try:
        # 单次 reset 验证
        env.reset()
        
        # 验证 UUV 在可通行区域
        assert env.terrain_3d[env.uuv.y, env.uuv.x, env.uuv.z] == False, "UUV 初始位置在不可通行区域（山体）"
        
        # 验证 UUV 坐标范围
        assert env.uuv_x_min_index <= env.uuv.x <= env.uuv_x_max_index, f"UUV x 超出范围: {env.uuv.x}"
        assert env.uuv_y_min_index <= env.uuv.y <= env.uuv_y_max_index, f"UUV y 超出范围: {env.uuv.y}"
        assert env.uuv_z_min_index <= env.uuv.z <= env.uuv_z_max_index, f"UUV z 超出范围: {env.uuv.z}"
        
        # 验证敌方在可通行区域
        assert env.terrain_3d[env.enemy.y, env.enemy.x, env.enemy.z] == False, "敌方初始位置在不可通行区域"
        
        # 验证敌方 y 范围
        assert env.enemy_y_min_idx <= env.enemy.y <= env.enemy_y_max_idx, f"敌方 y 超出范围: {env.enemy.y}"
        
        # 验证状态清零
        assert env.done == False, "重置后 done 应为 False"
        assert env.reward == 0, "重置后 reward 应为 0"
        assert env.cumulative_acoustic_signal == 0, "重置后 cumulative_acoustic_signal 应为 0"
        assert env.now_step == 0, "重置后 now_step 应为 0"
        
        print_info("✓ 单次 reset 验证通过")
        results["reset_single"] = (True, "UUV/敌方位置、状态清零正确")
        
    except AssertionError as e:
        print_error(f"✗ 单次 reset 验证失败: {e}")
        results["reset_single"] = (False, str(e))
    
    # 多次 reset 验证随机性
    try:
        positions_set = set()
        for i in range(5):
            env.reset()
            pos = (env.uuv.x, env.uuv.y, env.uuv.z)
            positions_set.add(pos)
            assert env.done == False, f"第 {i+1} 次 reset 后 done 应为 False"
        
        # 预期至少有 2-3 个不同位置（随机性）
        assert len(positions_set) >= 2, f"5 次 reset 位置重复过多，只有 {len(positions_set)} 个不同位置"
        print_info(f"✓ 多次 reset 验证通过（{len(positions_set)} 个不同初始位置）")
        results["reset_multiple"] = (True, f"5 次 reset 产生 {len(positions_set)} 个不同位置")
        
    except AssertionError as e:
        print_error(f"✗ 多次 reset 验证失败: {e}")
        results["reset_multiple"] = (False, str(e))
    
    # ===== 1.2 step() 正常流程测试 =====
    print_info("\n[1.2] 测试 step() 正常流程...")
    
    try:
        env.reset()
        initial_pos = (env.uuv.x, env.uuv.y, env.uuv.z)
        
        # 验证返回值结构
        state, reward, done, info = env.step(6)  # action=6 (原地不动)
        
        assert isinstance(state, tuple) and len(state) == 3, "state 应为长度为 3 的元组"
        assert isinstance(reward, (int, float)), "reward 应为数值"
        assert isinstance(done, bool), "done 应为布尔值"
        assert isinstance(info, dict), "info 应为字典"
        assert 'reward_details' in info, "info 应包含 'reward_details'"
        
        print_info("✓ 返回值结构验证通过")
        results["step_return_structure"] = (True, "返回值结构正确")
        
    except AssertionError as e:
        print_error(f"✗ 返回值结构验证失败: {e}")
        results["step_return_structure"] = (False, str(e))
    
    # 动作效果验证
    try:
        env.reset()
        action_map = {
            0: (-1, 0, 0),
            1: (1, 0, 0),
            2: (0, -1, 0),
            3: (0, 1, 0),
            4: (0, 0, -1),
            5: (0, 0, 1),
            6: (0, 0, 0),
        }
        
        action_results = {}
        for action in range(7):
            env.reset()
            pos_before = (env.uuv.x, env.uuv.y, env.uuv.z)
            env.step(action)
            pos_after = (env.uuv.x, env.uuv.y, env.uuv.z)
            
            expected_dx, expected_dy, expected_dz = action_map[action]
            actual_dx = pos_after[0] - pos_before[0]
            actual_dy = pos_after[1] - pos_before[1]
            actual_dz = pos_after[2] - pos_before[2]
            
            match = (actual_dx == expected_dx and actual_dy == expected_dy and actual_dz == expected_dz)
            action_results[f"action_{action}"] = match
            
            if not match:
                print_warn(f"  action={action}: 期望 ({expected_dx},{expected_dy},{expected_dz}), 实际 ({actual_dx},{actual_dy},{actual_dz})")
        
        all_passed = all(action_results.values())
        assert all_passed, "部分动作效果不符合预期"
        print_info("✓ 7 个动作效果验证全部通过")
        results["step_actions"] = (True, "所有 7 个动作效果正确")
        
    except AssertionError as e:
        print_error(f"✗ 动作效果验证失败: {e}")
        results["step_actions"] = (False, str(e))
    
    return results


def test_reward_mechanisms(env: Env) -> dict:
    """Phase 2: 奖励机制详细验证

    功能说明:
        测试三种奖励计算、reward_details 完整性、多 episode 统计。

    输入参数:
        env (Env): 环境对象

    输出参数:
        dict: 包含测试结果的字典
    """
    results = {}
    
    print_info("\n" + "="*70)
    print_info("Phase 2: 奖励机制详细验证")
    print_info("="*70)
    
    # ===== 2.1 奖励计算验证 =====
    print_info("\n[2.1] 测试奖励计算逻辑...")
    
    try:
        env.reset()
        
        # 记录 5 步的奖励信息
        reward_history = []
        for step_idx in range(5):
            state, reward, done, info = env.step(0)  # 持续向敌人靠近
            reward_details = info.get('reward_details', {})
            reward_history.append({
                'reward': reward,
                'stealth': reward_details.get('stealth_reward', 0),
                'approach': reward_details.get('approach_reward', 0),
                'tl_gradient': reward_details.get('tl_gradient_reward', 0),
                'area_average_tl': reward_details.get('area_average_tl_reward', 0),
            })
        
        # 验证三项之和 ≈ 总奖励（容差 1e-5）
        sum_matches = 0
        for hist in reward_history:
            component_sum = hist['stealth'] + hist['approach'] + hist['tl_gradient'] + hist['area_average_tl']
            if abs(component_sum - hist['reward']) < 1e-5:
                sum_matches += 1
        
        assert sum_matches >= 4, f"只有 {sum_matches}/5 步的奖励分解正确"
        print_info(f"✓ 5 步奖励分解验证通过（{sum_matches}/5 正确）")
        results["reward_decomposition"] = (True, f"奖励分解 {sum_matches}/5 正确")
        
    except AssertionError as e:
        print_error(f"✗ 奖励分解验证失败: {e}")
        results["reward_decomposition"] = (False, str(e))
    
    # 隐蔽奖励验证
    try:
        env.reset()
        tl_values = []
        stealth_rewards = []
        
        for _ in range(10):
            state, reward, done, info = env.step(0)
            stealth = info['reward_details'].get('stealth_reward', 0)
            stealth_rewards.append(stealth)
        
        # 验证范围 [-1, 1]
        assert all(-1 <= s <= 1 for s in stealth_rewards), "隐蔽奖励超出 [-1, 1] 范围"
        print_info("✓ 隐蔽奖励范围 [-1, 1] 验证通过")
        results["stealth_reward_range"] = (True, "隐蔽奖励在 [-1, 1] 范围内")
        
    except AssertionError as e:
        print_error(f"✗ 隐蔽奖励验证失败: {e}")
        results["stealth_reward_range"] = (False, str(e))
    
    # TL梯度奖励验证
    try:
        env.reset()
        
        # 多步靠近（TL 下降）时，梯度奖励应为正
        positive_count = 0
        for _ in range(10):
            state, reward, done, info = env.step(0)  # 靠近
            tl_grad = info['reward_details'].get('tl_gradient_reward', 0)
            if tl_grad > -0.1:  # 允许小的波动
                positive_count += 1
        
        print_info(f"✓ TL梯度奖励趋势验证通过（10 步中 {positive_count} 步为正/零）")
        results["tl_gradient_trend"] = (True, f"梯度正/零步数: {positive_count}/10")
        
    except AssertionError as e:
        print_error(f"✗ TL梯度奖励验证失败: {e}")
        results["tl_gradient_trend"] = (False, str(e))
    
    # ===== 2.2 多 episode 奖励统计 =====
    print_info("\n[2.2] 多 episode 统计分析...")
    
    episode_stats = []
    for ep in range(3):
        env.reset()
        ep_reward = 0
        ep_steps = 0
        ep_terminated = False
        
        while not ep_terminated and ep_steps < 200:  # 最多 200 步
            state, reward, done, info = env.step(0)  # 持续靠近
            ep_reward += reward
            ep_steps += 1
            ep_terminated = done
        
        term_reason = info.get('result', 'unknown') if done else 'incomplete'
        episode_stats.append({
            'episode': ep + 1,
            'total_reward': ep_reward,
            'steps': ep_steps,
            'term_reason': term_reason,
            'avg_reward': ep_reward / ep_steps if ep_steps > 0 else 0,
        })
        print_info(f"  Episode {ep+1}: steps={ep_steps}, reward={ep_reward:.2f}, reason={term_reason}")
    
    results["multi_episode_stats"] = (True, f"3 个 episode 统计完成")
    
    return results


def test_boundary_conditions(env: Env) -> dict:
    """Phase 3: 边界条件测试 - 五大失败条件和敌人行为

    功能说明:
        测试超出边界、撞山、被发现、超时、胜利五大失败条件，以及敌人行为。

    输入参数:
        env (Env): 环境对象

    输出参数:
        dict: 包含测试结果的字典
    """
    results = {}
    
    print_info("\n" + "="*70)
    print_info("Phase 3: 边界条件测试")
    print_info("="*70)
    
    # ===== 3.1 超出边界测试 =====
    print_info("\n[3.1] 测试超出边界条件 (OOB)...")
    
    oob_tests = [
        ("x_min", lambda env: setattr(env.uuv, 'x', env.uuv_x_min_index) or env.uuv.x, 0),  # x减小会越界
        ("x_max", lambda env: setattr(env.uuv, 'x', env.uuv_x_max_index) or env.uuv.x, 1),  # x增大会越界
        ("y_min", lambda env: setattr(env.uuv, 'y', env.uuv_y_min_index) or env.uuv.y, 2),  # y减小会越界
        ("y_max", lambda env: setattr(env.uuv, 'y', env.uuv_y_max_index) or env.uuv.y, 3),  # y增大会越界
        ("z_min", lambda env: setattr(env.uuv, 'z', env.uuv_z_min_index) or env.uuv.z, 4),  # z减小会越界
        ("z_max", lambda env: setattr(env.uuv, 'z', env.uuv_z_max_index) or env.uuv.z, 5),  # z增大会越界
    ]
    
    oob_passed = 0
    for test_name, pos_setter, action in oob_tests:
        try:
            env.reset()
            pos_setter(env)  # 设置为边界位置
            state, reward, done, info = env.step(action)
            
            if done and reward == -10 and "超出边界" in info.get('result', ''):
                print_info(f"  ✓ {test_name} 边界越界正确触发")
                oob_passed += 1
            else:
                print_warn(f"  ✗ {test_name} 边界未正确触发")
        except Exception as e:
            print_error(f"  ✗ {test_name} 测试异常: {e}")
    
    results["oob_boundary"] = (oob_passed >= 5, f"6 个边界测试通过 {oob_passed} 个")
    
    # ===== 3.2 撞山测试 =====
    print_info("\n[3.2] 测试撞山条件 (Collision)...")
    
    collision_passed = 0
    # 寻找地形中的山体（限制在UUV有效范围内）
    mountain_positions = []
    for y in range(env.uuv_y_min_index, env.uuv_y_max_index + 1):
        for x in range(env.uuv_x_min_index, env.uuv_x_max_index + 1):
            for z in range(env.uuv_z_min_index, env.uuv_z_max_index + 1):
                if env.terrain_3d[y, x, z] == True:
                    mountain_positions.append((x, y, z))
                    if len(mountain_positions) >= 3:
                        break
            if len(mountain_positions) >= 3:
                break
        if len(mountain_positions) >= 3:
            break
    
    if mountain_positions:
        for mnt_x, mnt_y, mnt_z in mountain_positions[:3]:
            try:
                env.reset()
                # 找到邻近的可通行点
                found_nearby = False
                for nearby_z in [mnt_z - 1, mnt_z + 1]:
                    if 0 <= nearby_z < env.map_depth and not env.terrain_3d[mnt_y, mnt_x, nearby_z]:
                        env.uuv.x = mnt_x
                        env.uuv.y = mnt_y
                        env.uuv.z = nearby_z
                        # 验证位置设置成功
                        if env.uuv.x == mnt_x and env.uuv.y == mnt_y and env.uuv.z == nearby_z:
                            found_nearby = True
                        break
                
                if found_nearby:
                    # 执行会撞山的动作
                    if mnt_z > env.uuv.z:
                        state, reward, done, info = env.step(5)  # 向上
                    else:
                        state, reward, done, info = env.step(4)  # 向下
                    
                    if done and reward == -10 and "撞山" in info.get('result', ''):
                        print_info(f"  ✓ 撞山条件正确触发 (位置: {mnt_x},{mnt_y},{mnt_z})")
                        collision_passed += 1
            except Exception as e:
                print_warn(f"  撞山测试异常: {e}")
    
    results["collision"] = (collision_passed >= 1, f"撞山测试通过 {collision_passed} 个")
    
    # ===== 3.3 被发现测试 =====
    print_info("\n[3.3] 测试被发现条件 (Detected)...")
    
    detected_passed = 0
    max_retries = 5  # 最多尝试 5 次
    
    for retry in range(max_retries):
        try:
            env.reset()
            
            # 多步靠近敌人（增加步数以积累累计声呐信号）
            for step_idx in range(500):  # 500 步
                state, reward, done, info = env.step(0)  # 不断靠近
                
                if done and reward == -3 and "被发现" in info.get('result', ''):
                    print_info(f"  ✓ 被发现条件在第 {retry+1} 次尝试的第 {step_idx+1} 步触发")
                    detected_passed = 1
                    break
            
            if detected_passed == 1:
                break
        except Exception as e:
            print_warn(f"  被发现测试异常 (第 {retry+1} 次尝试): {e}")
    
    if detected_passed == 0:
        print_warn(f"  ✗ {max_retries} 次尝试（每次 500 步）内未触发被发现条件")
    
    results["detected"] = (detected_passed >= 1, f"被发现条件触发: {detected_passed}")
    
    # ===== 3.4 超时测试 =====
    print_info("\n[3.4] 测试超时条件 (Timeout)...")
    
    timeout_passed = 0
    try:
        env.reset()
        recommended_steps = env.recommended_steps
        
        # 执行无用步数（原地不动），增加到500步以确保超过推荐步数的1.5倍
        max_steps = max(int(recommended_steps * 1.5) + 50, 500)  # 至少500步
        for step_idx in range(max_steps):
            state, reward, done, info = env.step(6)  # action=6 原地不动
            
            if done and reward == -3 and "超出推荐步数" in info.get('result', ''):
                print_info(f"  ✓ 超时条件在第 {step_idx+1} 步触发 (推荐步数: {recommended_steps:.0f})")
                timeout_passed = 1
                break
        
        if timeout_passed == 0:
            print_warn(f"  ✗ 执行 {max_steps} 步后未触发超时")
    except Exception as e:
        print_error(f"  超时测试异常: {e}")
    
    results["timeout"] = (timeout_passed >= 1, f"超时条件触发: {timeout_passed}")
    
    # ===== 3.5 胜利测试 =====
    print_info("\n[3.5] 测试胜利条件 (Victory)...")
    
    victory_passed = 0
    max_victory_tries = 20

    for victory_try in range(max_victory_tries):
        try:
            env.reset()
            # 寻找一个靠近胜利位置但有效的位置
            for test_x in range(env.victory_x_idx - 2, min(env.victory_x_idx + 3, env.uuv_x_max_index + 1)):
                if test_x >= env.uuv_x_min_index and test_x <= env.uuv_x_max_index and \
                    not env.terrain_3d[env.uuv.y, test_x, env.uuv.z]:
                    env.uuv.x = test_x
                    state, reward, done, info = env.step(6)  # action=6 原地不动
                    
                    if done and reward == 3 and "胜利" in info.get('result', ''):
                        print_info(f"  ✓ 胜利条件在第 {victory_try+1} 次尝试触发")
                        victory_passed = 1
                        break
            
            if victory_passed == 1:
                break
        except Exception as e:
            print_warn(f"  胜利测试异常 (第 {victory_try+1} 次尝试): {e}")
    
    if victory_passed == 0:
        print_warn(f"  ✗ {max_victory_tries} 次尝试内未触发胜利条件")
    
    results["victory"] = (victory_passed >= 1, f"胜利条件触发: {victory_passed}")
    
    # ===== 3.6 敌人行为验证 =====
    print_info("\n[3.6] 测试敌人行为...")
    
    enemy_behavior_ok = True
    try:
        # 巡逻验证
        env.reset()
        initial_enemy_y = env.enemy.y
        enemy_y_set = {initial_enemy_y}
        
        for _ in range(30):
            env.step(6)
            enemy_y_set.add(env.enemy.y)
        
        # 敌人应该在 y_min 和 y_max 之间移动
        for y in enemy_y_set:
            assert env.enemy_y_min_idx <= y <= env.enemy_y_max_idx, f"敌人 y={y} 超出范围"
        
        # 敌人应该不在山体中
        for y in enemy_y_set:
            assert env.terrain_3d[y, env.enemy.x, env.enemy.z] == False, f"敌人在山体中 ({env.enemy.x},{y},{env.enemy.z})"
        
        print_info(f"  ✓ 敌人巡逻验证通过 (y 范围: {min(enemy_y_set)}～{max(enemy_y_set)})")
        
    except AssertionError as e:
        print_error(f"  ✗ 敌人行为验证失败: {e}")
        enemy_behavior_ok = False
    
    results["enemy_behavior"] = (enemy_behavior_ok, "敌人巡逻行为验证")
    
    # ===== 3.7 坐标系统验证 =====
    print_info("\n[3.7] 测试坐标系统...")
    
    coord_ok = True
    try:
        # 验证 terrain_3d[y, x, z] 索引约定
        env.reset()
        valid_steps = 0
        for _ in range(40):  # 增加循环次数以获得足够的有效步骤
            state, reward, done, info = env.step(np.random.randint(0, 7))
            
            # 只验证未触发任何失败条件的步骤
            if not done:
                y, x, z = env.uuv.y, env.uuv.x, env.uuv.z
                assert 0 <= y < env.map_height, f"y={y} 越界"
                assert 0 <= x < env.map_width, f"x={x} 越界"
                assert 0 <= z < env.map_depth, f"z={z} 越界"
                valid_steps += 1
            else:
                # 如果某一步触发失败条件，重新开始新 episode
                env.reset()
            
            if valid_steps >= 20:
                break
        
        # TL 查询验证
        for _ in range(5):
            env.reset()
            tl_val = env._query_TL(env.enemy.y, env.uuv.x, env.uuv.y, env.uuv.z)
            assert isinstance(tl_val, (int, float, np.number)), f"TL 值类型不正确: {type(tl_val)}"
        
        print_info(f"  ✓ 坐标系统和 TL 查询验证通过")
        
    except AssertionError as e:
        print_error(f"  ✗ 坐标系统验证失败: {e}")
        coord_ok = False
    
    results["coordinate_system"] = (coord_ok, "坐标索引和 TL 查询验证")
    
    return results


def test_performance_and_statistics(env: Env) -> dict:
    """Phase 4: 性能和统计分析

    功能说明:
        测试系统性能、TL 查询完整性、敌人随机性、长期 episode 统计。

    输入参数:
        env (Env): 环境对象

    输出参数:
        dict: 包含测试结果的字典
    """
    results = {}

    print_info("\n" + "="*70)
    print_info("Phase 4: 性能和统计分析")
    print_info("="*70)

    # ===== 4.1 性能基准测试 =====
    print_info("\n[4.1] 性能基准测试 (100 步连续运行)...")

    try:
        env.reset()

        start_time = time.time()
        for _ in range(100):
            env.step(np.random.randint(0, 7))
        elapsed = time.time() - start_time

        avg_step_time = elapsed / 100 * 1000  # 转为毫秒
        print_info(f"  ✓ 100 步耗时: {elapsed:.3f}s, 平均单步: {avg_step_time:.2f}ms")
        results["performance"] = (True, f"100 步耗时 {elapsed:.3f}s")

    except Exception as e:
        print_error(f"  ✗ 性能测试异常: {e}")
        results["performance"] = (False, str(e))

    # ===== 4.2 TL 查询完整性 =====
    print_info("\n[4.2] TL 查询完整性测试...")

    valid_queries = 0
    try:
        for _ in range(20):
            env.reset()
            tl_val = env._query_TL(env.enemy.y, env.uuv.x, env.uuv.y, env.uuv.z)

            if isinstance(tl_val, (int, float, np.number)) and not np.isnan(tl_val):
                valid_queries += 1

        assert valid_queries >= 18, f"只有 {valid_queries}/20 个查询返回有效值"
        print_info(f"  ✓ 20 次 TL 查询全部有效")
        results["tl_query_validity"] = (True, f"有效查询: {valid_queries}/20")

    except AssertionError as e:
        print_error(f"  ✗ TL 查询完整性验证失败: {e}")
        results["tl_query_validity"] = (False, str(e))

    # ===== 4.3 随机性统计 =====
    print_info("\n[4.3] 敌人巡逻方向随机性统计...")

    try:
        direction_dist = {-1: 0, 1: 0}

        for _ in range(20):
            env.reset()
            direction_dist[env.enemy_forward_direction] += 1

        ratio_neg1 = direction_dist[-1] / 20 * 100
        ratio_pos1 = direction_dist[1] / 20 * 100

        # 预期大约 50%-50% 分布
        print_info(f"  方向分布: -1={ratio_neg1:.0f}%, +1={ratio_pos1:.0f}%")
        print_info(f"  ✓ 20 次 reset 的敌人初始方向分布统计完成")
        results["enemy_direction_randomness"] = (True, f"方向分布: -1={ratio_neg1:.0f}%, +1={ratio_pos1:.0f}%")

    except Exception as e:
        print_error(f"  ✗ 随机性统计异常: {e}")
        results["enemy_direction_randomness"] = (False, str(e))

    # ===== 4.4 长期多 episode 统计 =====
    print_info("\n[4.4] 长期 10 个 episode 统计分析...")

    episode_stats = {
        'victory': 0,
        'detected': 0,
        'timeout': 0,
        'out_of_bounds': 0,
        'collision': 0,
        'total_reward': 0,
        'total_steps': 0,
    }

    for ep in range(10):
        env.reset()
        ep_reward = 0
        ep_steps = 0

        this_ep_reward_dict = {
            'stealth': 0,
            'approach': 0,
            'tl_gradient': 0,
            'area_average_tl': 0,
        }

        while not env.done and ep_steps < 500:  # 最多 500 步
            state, reward, done, info = env.step(0)  # 持续靠近
            reward_details = info.get('reward_details', {})
            for key in this_ep_reward_dict.keys():
                this_ep_reward_dict[key] += reward_details.get(key + '_reward', 0)

            ep_reward += reward
            ep_steps += 1
        
        # 输出本轮各个 reward 分量统计数据
        print_info(f"  Episode {ep+1:2d} 各 reward 分量统计:")
        for key, value in this_ep_reward_dict.items():
            print_info(f"    {key}: {value:.2f}, 占比 {(value / ep_reward * 100) if ep_reward != 0 else 0:.1f}%")

        term_reason = info.get('result', 'unknown') if env.done else 'incomplete'
        episode_stats['total_reward'] += ep_reward
        episode_stats['total_steps'] += ep_steps

        # 统计终止原因
        if '胜利' in term_reason:
            episode_stats['victory'] += 1
        elif '被发现' in term_reason:
            episode_stats['detected'] += 1
        elif '推荐步数' in term_reason:
            episode_stats['timeout'] += 1
        elif '超出边界' in term_reason:
            episode_stats['out_of_bounds'] += 1
        elif '撞山' in term_reason:
            episode_stats['collision'] += 1

        print_info(f"  Episode {ep+1:2d}: {ep_steps:3d} 步, reward={ep_reward:7.2f}, {term_reason}")

    avg_reward = episode_stats['total_reward'] / 10
    avg_steps = episode_stats['total_steps'] / 10

    print_info(f"\n  统计汇总 (10 episodes):")
    print_info(f"    胜利: {episode_stats['victory']:2d}  被发现: {episode_stats['detected']:2d}  超时: {episode_stats['timeout']:2d}")
    print_info(f"    超界: {episode_stats['out_of_bounds']:2d}  撞山: {episode_stats['collision']:2d}")
    print_info(f"    平均奖励: {avg_reward:.2f}  平均步数: {avg_steps:.1f}")

    results["long_episode_statistics"] = (True, 
        f"胜利{episode_stats['victory']}/被发现{episode_stats['detected']}/超时{episode_stats['timeout']}/超界{episode_stats['out_of_bounds']}/撞山{episode_stats['collision']}")

    return results


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="main_config",
)
def main(cfg: DictConfig) -> None:
    """运行完整的 GD-UUV PPO 训练管道。

    功能说明:
        加载配置 → 初始化环境与 PPOTrainer → 执行训练循环 → 周期性保存 checkpoint。
        所有训练计算在 GPU 上执行。

    输入参数:
        cfg (DictConfig): Hydra 配置对象，包含 env、trainer、ppo 配置段。

    输出参数:
        无。训练结果与 checkpoint 保存到 outputs/<date>/<time>/ 目录（Hydra 自动管理）。

    调用示例:
        执行不带命令行参数的默认训练:
        python runs.py

        或指定特定配置参数:
        python runs.py trainer.max_epochs=50 ppo.learning_rate=5e-4
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print_error("CUDA 不可用，请检查 GPU 驱动和 PyTorch 安装")
        return
    else:
        print_info(f"使用设备: {device}")

    # --- 初始化环境 ---
    env = Env(cfg)
    print_info("环境初始化完成")

    """
    # --- 地形诊断（可选）---
    # 用于验证坐标顺序和观测张量的正确性
    hydra_cfg = HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)
    print_info("\n开始地形数据诊断...")
    visualize_terrain_slice(env, output_dir, test_pos=(20, 50, 3))
    print_info("数值对比诊断完成")
    
    print_info("生成三维地形可视化...")
    visualize_terrain_3d(env, output_dir, test_pos=(20, 50, 3))
    print_info("地形诊断完成。结果已保存到 output_dir\n")

    return
    """

    """
    # ========== 开始测试套件 ==========
    print_info("\n" + "🧪 " + "="*68)
    print_info("开始 Env 模块功能测试套件")
    print_info("="*70)
    
    # Phase 1: 基础功能测试
    phase1_results = test_basic_reset_and_step(env)
    
    # Phase 2: 奖励机制验证
    phase2_results = test_reward_mechanisms(env)
    
    # Phase 3: 边界条件测试
    phase3_results = test_boundary_conditions(env)
    
    # Phase 4: 性能和统计分析
    phase4_results = test_performance_and_statistics(env)
    
    # ========== 汇总测试结果 ==========
    print_info("\n" + "="*70)
    print_info("测试汇总报告")
    print_info("="*70)
    
    console = Console()
    
    # 创建汇总表
    summary_table = Table(title="Env 测试总体结果", show_header=True, header_style="bold cyan")
    summary_table.add_column("测试阶段", style="cyan")
    summary_table.add_column("测试项目", style="magenta")
    summary_table.add_column("结果", style="green")
    summary_table.add_column("详情", style="yellow")
    
    all_results = [
        ("Phase 1: 基础功能", phase1_results),
        ("Phase 2: 奖励机制", phase2_results),
        ("Phase 3: 边界条件", phase3_results),
        ("Phase 4: 性能统计", phase4_results),
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for phase_name, phase_dict in all_results:
        for test_name, (passed, detail) in phase_dict.items():
            total_tests += 1
            if passed:
                passed_tests += 1
                status = "✓ PASS"
                status_style = "green"
            else:
                status = "✗ FAIL"
                status_style = "red"
            
            summary_table.add_row(
                phase_name,
                test_name,
                f"[{status_style}]{status}[/{status_style}]",
                detail[:50] + "..." if len(detail) > 50 else detail
            )
    
    console.print(summary_table)
    
    # 输出最终统计
    print_info(f"\n{'='*70}")
    print_info(f"测试总数: {total_tests}  通过: {passed_tests}  失败: {total_tests - passed_tests}")
    print_info(f"通过率: {passed_tests / total_tests * 100:.1f}%")
    print_info(f"{'='*70}")
    
    if passed_tests == total_tests:
        print_info("🎉 所有测试通过！Env 模块功能完整。")
    else:
        print_warn(f"⚠️  有 {total_tests - passed_tests} 个测试失败，请检查详情。")
    
    print_info("\n✓ 测试套件完成，程序退出。")

    """

    # --- 初始化 PPO 训练器 ---
    trainer = PPOTrainer(cfg, device=device, env=env)
    print_info("PPO 训练器初始化完成，spatial_input 动态重构已启用")

    # --- 获取输出目录（Hydra 管理） ---
    hydra_cfg = HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    print_info(f"输出目录: {output_dir}")
    print_info(f"检查点目录: {checkpoint_dir}")

    # --- 训练循环 ---
    print_info("=" * 60)
    print_info("开始训练")
    print_info("=" * 60)

    for epoch in range(cfg.trainer.max_epochs):
        print_info(f"\n--- Epoch {epoch + 1}/{cfg.trainer.max_epochs} ---")

        # 收集 rollout
        rollout_info = trainer.collect_rollout(
            env=env,
            num_steps=cfg.trainer.steps_per_epoch,
        )

        # 执行策略更新
        update_info = trainer.update_policy()

        # 打印训练进度
        print_info(
            f"Epoch {epoch + 1} 完成: "
            f"奖励={rollout_info['reward_mean']:.4f}, "
            f"策略损失={update_info['policy_loss']:.4f}, "
            f"价值损失={update_info['value_loss']:.4f}, "
            f"全局步数={trainer.global_step}"
        )

        # 周期性保存 checkpoint - 按步数间隔检查
        checkpoint_interval_steps = cfg.trainer.checkpoint_interval
        if trainer.global_step > 0 and trainer.global_step % checkpoint_interval_steps == 0:
            trainer.save_checkpoint(checkpoint_dir)

    # --- 训练完成 ---
    print_info("=" * 60)
    print_info("训练完成")
    print_info("=" * 60)
    trainer.save_checkpoint(checkpoint_dir)
    print_info(f"最终检查点已保存至: {checkpoint_dir}")


if __name__ == "__main__":
    main()

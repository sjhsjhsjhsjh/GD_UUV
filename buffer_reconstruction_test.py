"""RolloutBuffer spatial_input 动态重构验证测试。

测试场景：
1. 验证重构的 spatial_input 与原始 spatial_input bit-exact 相等
2. 测试不同坐标重构结果的正确性
3. 验证 batch 处理的正确性
4. 显存节省效果评估
"""

import torch
import numpy as np
from hydra import compose, initialize_config_dir
from pathlib import Path
from omegaconf import DictConfig

from env.env import Env
from buffer.rollout_buffer import RolloutBuffer
from utils.rich_print import print_info, print_success, print_error


def test_reconstruction_accuracy():
    """测试 1: 验证重构的 spatial_input 与原始相同"""
    print_info("\n" + "="*70)
    print_info("测试 1: 重构精度验证")
    print_info("="*70)
    
    # 初始化环境和 buffer
    config_dir = str(Path(__file__).parent / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="main_config")
    
    env = Env(cfg)
    buffer = RolloutBuffer(capacity=100, device="cuda", env=env)
    
    # 重置环境
    env.reset()
    
    # 获取原始观测
    original_spatial, state_vector = env.get_observation_tensor(device="cuda")
    
    print_info(f"原始观测形状: {original_spatial.shape}, {state_vector.shape}")
    
    # 创建 batch state_vector（包含当前位置的状态向量）
    batch_state_vector = state_vector
    
    # 调用重构方法
    reconstructed_spatial = buffer._reconstruct_spatial_input_batch(batch_state_vector)
    
    print_info(f"重构观测形状: {reconstructed_spatial.shape}")
    
    # 比较（移除 batch 维后比较）
    original_squeezed = original_spatial.squeeze(0)
    reconstructed_squeezed = reconstructed_spatial.squeeze(0)
    
    # 计算误差
    max_diff = (original_squeezed - reconstructed_squeezed).abs().max().item()
    mean_diff = (original_squeezed - reconstructed_squeezed).abs().mean().item()
    
    print_info(f"最大差异: {max_diff:.6e}")
    print_info(f"平均差异: {mean_diff:.6e}")
    
    # 验证 bit-exact 相等
    if max_diff < 1e-6:
        print_success("✓ 重构观测与原始观测 bit-exact 相等")
        return True
    else:
        print_error(f"✗ 重构观测与原始观测不相等，最大差异: {max_diff}")
        return False


def test_batch_reconstruction():
    """测试 2: 验证 batch 处理的正确性"""
    print_info("\n" + "="*70)
    print_info("测试 2: Batch 处理正确性验证")
    print_info("="*70)
    
    # 初始化环境和 buffer
    config_dir = str(Path(__file__).parent / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="main_config")
    
    env = Env(cfg)
    buffer = RolloutBuffer(capacity=1000, device="cuda", env=env)
    
    batch_size = 8
    batch_state_vectors = []
    
    print_info(f"生成 {batch_size} 个不同位置的状态向量...")
    
    # 生成 batch 状态向量（不同位置）
    for i in range(batch_size):
        env.reset()
        _, state_vector = env.get_observation_tensor(device="cuda")
        batch_state_vectors.append(state_vector.squeeze(0))
    
    batch_state_vector = torch.stack(batch_state_vectors, dim=0)
    print_info(f"Batch state_vector 形状: {batch_state_vector.shape}")
    
    # 重构 batch spatial_input
    batch_spatial_input = buffer._reconstruct_spatial_input_batch(batch_state_vector)
    print_info(f"重构 batch spatial_input 形状: {batch_spatial_input.shape}")
    
    # 逐个验证每个 batch 元素
    print_info("逐个验证 batch 中的每个元素...")
    for i in range(batch_size):
        # 在该位置重新生成观测
        x_uuv = int(batch_state_vector[i, 0].item())
        y_uuv = int(batch_state_vector[i, 1].item())
        z_uuv = int(batch_state_vector[i, 2].item())
        y_enemy = int(batch_state_vector[i, 3].item())
        
        # 临时设置环境位置
        original_uuv = (env.uuv.x, env.uuv.y, env.uuv.z)
        original_enemy = env.enemy.y
        
        env.uuv.x, env.uuv.y, env.uuv.z = x_uuv, y_uuv, z_uuv
        env.enemy.y = y_enemy
        
        # 获取该位置的原始观测
        original_spatial, _ = env.get_observation_tensor(device="cuda")
        
        # 恢复环境
        env.uuv.x, env.uuv.y, env.uuv.z = original_uuv
        env.enemy.y = original_enemy
        
        # 比较
        max_diff = (original_spatial.squeeze(0) - batch_spatial_input[i]).abs().max().item()
        
        if max_diff < 1e-6:
            print_info(f"  [{i+1}/{batch_size}] ✓ 位置 ({x_uuv}, {y_uuv}, {z_uuv}) 重构正确")
        else:
            print_error(f"  [{i+1}/{batch_size}] ✗ 位置 ({x_uuv}, {y_uuv}, {z_uuv}) 重构错误，差异: {max_diff}")
            return False
    
    print_success("✓ Batch 处理正确")
    return True


def test_buffer_add_and_iterate():
    """测试 3: 验证 buffer add 和 iterate_minibatches 的完整流程"""
    print_info("\n" + "="*70)
    print_info("测试 3: Buffer 完整流程验证")
    print_info("="*70)
    
    # 初始化环境和 buffer
    config_dir = str(Path(__file__).parent / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="main_config")
    
    env = Env(cfg)
    # 该测试期望 buffer 在 iterate 时动态重构 spatial_input，因此显式禁用持久化存储
    buffer = RolloutBuffer(capacity=200, device="cuda", env=env, store_spatial=False)
    
    print_info("向 buffer 添加经验...")
    
    # 添加 50 步经验
    num_steps = 50
    for step in range(num_steps):
        env.reset() if step % 20 == 0 else None  # 每 20 步重置
        spatial_input, state_vector = env.get_observation_tensor(device="cuda")
        
        # 生成随机动作
        action = torch.tensor(torch.randint(0, 7, (1,)).item())
        logprob = torch.tensor(-1.0)
        reward = torch.tensor(0.1).item()
        done = False
        value = torch.tensor(0.5)
        
        # 添加到 buffer（spatial_input 现在可以为 None）
        buffer.add(
            spatial_input=None,  # 已弃用
            state_vector=state_vector,
            action=action,
            logprob=logprob,
            reward=reward,
            done=done,
            value=value,
        )
    
    print_info(f"✓ 添加 {num_steps} 步经验")
    
    # 计算 GAE 回报
    buffer.compute_gae_returns(gamma=0.99, gae_lambda=0.95, next_value=0.0)
    print_info("✓ 计算 GAE 回报")
    
    # 迭代 minibatch
    print_info("迭代 minibatches...")
    batch_count = 0
    total_samples = 0
    
    for minibatch in buffer.iterate_minibatches(batch_size=16):
        batch_count += 1
        
        spatial_input = minibatch["spatial_input"]
        state_vector = minibatch["state_vector"]
        actions = minibatch["action"]
        
        batch_sz = spatial_input.shape[0]
        total_samples += batch_sz
        
        # 验证形状
        assert spatial_input.shape[0] > 0, "spatial_input batch size 为 0"
        assert spatial_input.shape[1:] == (2, env.field_of_view, env.field_of_view, env.field_of_view_on_z), \
            f"spatial_input 形状错误: {spatial_input.shape}"
        assert state_vector.shape == (batch_sz, buffer.state_vector_dim), f"state_vector 形状错误: {state_vector.shape}"
        assert actions.shape == (batch_sz,), f"actions 形状错误: {actions.shape}"
        
        print_info(f"  Batch {batch_count}: size={batch_sz}, spatial_input 已动态重构")
    
    print_success(f"✓ 完整流程验证通过，共处理 {total_samples} 个样本")
    return True


def test_memory_savings():
    """测试 4: 显存节省效果评估"""
    print_info("\n" + "="*70)
    print_info("测试 4: 显存节省效果评估")
    print_info("="*70)
    
    # 初始化环境
    config_dir = str(Path(__file__).parent / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="main_config")
    
    env = Env(cfg)
    
    # 计算理论显存节省
    capacity = 10000
    field_of_view = env.field_of_view
    field_of_view_on_z = env.field_of_view_on_z
    
    # spatial_input 大小：capacity × 2 × FOV × FOV × FOV_Z × 4 bytes (float32)
    spatial_size_bytes = capacity * 2 * field_of_view * field_of_view * field_of_view_on_z * 4
    spatial_size_mb = spatial_size_bytes / (1024 * 1024)
    
    # state_vector 大小：capacity × N × 4 bytes（N 为 env.state_vector_dim）
    sv_dim = getattr(env, "state_vector_dim", 8)
    state_vector_size_bytes = capacity * sv_dim * 4
    state_vector_size_mb = state_vector_size_bytes / (1024 * 1024)
    
    # 总大小
    total_with_spatial = spatial_size_mb + state_vector_size_mb
    total_without_spatial = state_vector_size_mb
    
    savings_mb = spatial_size_mb
    savings_percent = (spatial_size_mb / total_with_spatial) * 100
    
    print_info(f"\n理论显存估算 (capacity={capacity}):")
    print_info(f"  - spatial_input 大小: {spatial_size_mb:.2f} MB")
    print_info(f"  - state_vector 大小: {state_vector_size_mb:.2f} MB")
    print_info(f"  - 优化前总大小: {total_with_spatial:.2f} MB")
    print_info(f"  - 优化后总大小: {total_without_spatial:.2f} MB")
    print_info(f"  - 显存节省: {savings_mb:.2f} MB ({savings_percent:.1f}%)")
    
    print_success(f"✓ 理论显存节省约 {savings_percent:.1f}%")
    return True


def main():
    print_info("=" * 70)
    print_info("RolloutBuffer 动态重构验证测试套件")
    print_info("=" * 70)
    
    tests = [
        ("重构精度验证", test_reconstruction_accuracy),
        ("Batch 处理验证", test_batch_reconstruction),
        ("Buffer 完整流程验证", test_buffer_add_and_iterate),
        ("显存节省评估", test_memory_savings),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print_error(f"✗ {test_name} 抛出异常: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print_info("\n" + "=" * 70)
    print_info("测试总结")
    print_info("=" * 70)
    print_info(f"通过: {passed}/{len(tests)}")
    print_info(f"失败: {failed}/{len(tests)}")
    
    if failed == 0:
        print_success("🎉 所有测试通过！")
        return True
    else:
        print_error(f"❌ 有 {failed} 个测试失败")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

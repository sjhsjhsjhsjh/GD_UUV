"""冒烟测试：验证 trainer + buffer 整个管道的初始化与基本功能。

运行方式:
    E:/lib/conda-env/torch_gpu/python.exe e:/program/GD-UUV-self/smoke_test.py
"""

import sys
import torch
import numpy as np
from pathlib import Path
from omegaconf import DictConfig

from utils.rich_print import print_info, print_error, print_success
from env.env import Env
from agent.acnet import ACNet
from agent.trainer import PPOTrainer
from buffer.rollout_buffer import RolloutBuffer


def test_imports() -> bool:
    """测试所有必要模块的导入。"""
    print_info("=" * 60)
    print_info("测试 1: 导入检查")
    print_info("=" * 60)
    
    try:
        print_info("✓ ACNet 导入成功")
        print_info("✓ RolloutBuffer 导入成功")
        print_info("✓ Env 导入成功")
        print_info("✓ PPOTrainer 导入成功")
        print_success("导入检查通过")
        return True
    except Exception as e:
        print_error(f"导入检查失败: {e}")
        return False


def test_device_availability() -> bool:
    """测试 GPU 可用性。"""
    print_info("=" * 60)
    print_info("测试 2: 设备检查")
    print_info("=" * 60)
    
    if not torch.cuda.is_available():
        print_error("CUDA 不可用！trainer 需要 GPU。")
        return False
    
    device = torch.device("cuda")
    print_info(f"✓ CUDA 可用，设备: {device}")
    print_info(f"✓ GPU 数量: {torch.cuda.device_count()}")
    print_info(f"✓ 当前 GPU: {torch.cuda.get_device_name(0)}")
    print_success("设备检查通过")
    return True


def test_acnet_initialization() -> bool:
    """测试 ACNet 初始化与前向传播。"""
    print_info("=" * 60)
    print_info("测试 3: ACNet 初始化与推理")
    print_info("=" * 60)
    
    try:
        device = "cuda"
        model = ACNet(device=device).to(device)
        print_info("✓ ACNet 初始化成功")
        
        # 测试前向传播
        batch_size, depth, height, width = 2, 16, 16, 16
        spatial_input = torch.randn(batch_size, 2, depth, height, width, device=device)
        state_vector = torch.randn(batch_size, 6, device=device)
        
        actor_logits, state_value = model(spatial_input, state_vector)
        
        assert actor_logits.shape == (batch_size, 6), f"Actor logits 形状错误: {actor_logits.shape}"
        assert state_value.shape == (batch_size, 1), f"State value 形状错误: {state_value.shape}"
        assert actor_logits.device.type == "cuda", "Actor logits 不在 GPU 上"
        assert state_value.device.type == "cuda", "State value 不在 GPU 上"
        
        print_info(f"✓ 前向传播成功")
        print_info(f"  - Actor logits: {tuple(actor_logits.shape)}, device={actor_logits.device}")
        print_info(f"  - State value: {tuple(state_value.shape)}, device={state_value.device}")
        print_success("ACNet 初始化与推理通过")
        return True
    except Exception as e:
        print_error(f"ACNet 测试失败: {e}")
        return False


def test_rollout_buffer() -> bool:
    """测试 RolloutBuffer 的基本操作。"""
    print_info("=" * 60)
    print_info("测试 4: RolloutBuffer 操作")
    print_info("=" * 60)
    
    try:
        device = "cuda"
        buffer = RolloutBuffer(capacity=100, device=device)
        print_info("✓ RolloutBuffer 初始化成功")
        
        # 模拟几步轨迹
        for step in range(5):
            spatial_input = torch.randn(1, 2, 16, 16, 16, device=device)
            state_vector = torch.randn(1, 6, device=device)
            action = torch.tensor([step % 6], device=device)
            logprob = torch.tensor(-0.5, device=device)
            reward = float(step) + 1.0
            done = (step == 4)
            value = torch.tensor(0.5, device=device)
            
            buffer.add(spatial_input, state_vector, action, logprob, reward, done, value)
        
        print_info(f"✓ 添加 5 步经验成功，ptr={buffer.ptr}")
        
        # 计算 GAE 返回值
        buffer.compute_gae_returns(gamma=0.99, gae_lambda=0.95, next_value=0.0)
        print_info("✓ GAE 计算成功")
        print_info(f"  - Returns shape: {buffer.returns.shape}")
        print_info(f"  - Advantages shape: {buffer.advantages.shape}")
        
        # 迭代 minibatch
        num_minibatches = 0
        for minibatch in buffer.iterate_minibatches(batch_size=2):
            num_minibatches += 1
            assert minibatch["spatial_input"].shape[0] <= 2
            assert minibatch["state_vector"].shape[0] <= 2
            assert minibatch["action"].shape[0] <= 2
        
        print_info(f"✓ Minibatch 迭代成功，共 {num_minibatches} 个 batch")
        
        # 清空 buffer
        buffer.clear()
        assert buffer.ptr == 0
        print_info("✓ Buffer 清空成功")
        
        print_success("RolloutBuffer 操作通过")
        return True
    except Exception as e:
        print_error(f"RolloutBuffer 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_env_observation_generation() -> bool:
    """测试环境观测生成。"""
    print_info("=" * 60)
    print_info("测试 5: 环境观测生成")
    print_info("=" * 60)
    
    try:
        # 加载配置
        from hydra import compose, initialize_config_dir
        config_dir = str(Path(__file__).parent / "configs")
        
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="main_config")
        
        # 初始化环境
        env = Env(cfg)
        print_info("✓ 环境初始化成功")
        
        # 重置环境
        env.reset()
        print_info("✓ 环境重置成功")
        
        # 获取观测张量
        device = "cuda"
        spatial_input, state_vector = env.get_observation_tensor(device=device, window_size=16)
        
        assert spatial_input.shape == (1, 2, 16, 16, 16), f"Spatial 形状错误: {spatial_input.shape}"
        assert state_vector.shape == (1, 6), f"State vector 形状错误: {state_vector.shape}"
        assert spatial_input.device.type == "cuda", "Spatial 不在 GPU 上"
        assert state_vector.device.type == "cuda", "State vector 不在 GPU 上"
        
        print_info(f"✓ 观测生成成功")
        print_info(f"  - Spatial input: {tuple(spatial_input.shape)}, device={spatial_input.device}")
        print_info(f"  - State vector: {tuple(state_vector.shape)}, device={state_vector.device}")
        
        # 执行一步
        action = 0
        next_pos, reward, done, info = env.step(action)
        print_info(f"✓ 环境 step 执行成功: reward={reward:.4f}, done={done}")
        
        print_success("环境观测生成通过")
        return True
    except Exception as e:
        print_error(f"环境观测生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_initialization() -> bool:
    """测试 PPOTrainer 初始化。"""
    print_info("=" * 60)
    print_info("测试 6: PPOTrainer 初始化")
    print_info("=" * 60)
    
    try:
        from hydra import compose, initialize_config_dir
        config_dir = str(Path(__file__).parent / "configs")
        
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="main_config")
        
        device = "cuda"
        trainer = PPOTrainer(cfg, device=device)
        print_info("✓ PPOTrainer 初始化成功")
        print_info(f"  - 模型参数数: {sum(p.numel() for p in trainer.model.parameters())}")
        print_info(f"  - 学习率: {cfg.ppo.learning_rate}")
        print_info(f"  - Gamma: {cfg.ppo.gamma}")
        
        print_success("PPOTrainer 初始化通过")
        return True
    except Exception as e:
        print_error(f"PPOTrainer 初始化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有冒烟测试。"""
    print_info("=" * 60)
    print_info("GD-UUV Trainer 冒烟测试开始")
    print_info("=" * 60 + "\n")
    
    tests = [
        ("导入检查", test_imports),
        ("设备检查", test_device_availability),
        ("ACNet 初始化", test_acnet_initialization),
        ("RolloutBuffer 操作", test_rollout_buffer),
        ("环境观测生成", test_env_observation_generation),
        ("PPOTrainer 初始化", test_trainer_initialization),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_error(f"测试 '{test_name}' 异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
        
        print()
    
    # 总结
    print_info("=" * 60)
    print_info("测试总结")
    print_info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print_info(f"{status}: {test_name}")
    
    print_info(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print_success("所有冒烟测试通过！整个训练管道可用。")
        return 0
    else:
        print_error(f"部分测试失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

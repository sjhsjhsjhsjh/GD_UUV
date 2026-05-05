import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pathlib import Path
import torch

from env.env import Env
from agent.trainer import PPOTrainer
from utils.rich_print import print_info, print_warn, print_error


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
    # --- 初始化 ---
    print_info("=" * 60)
    print_info("GD-UUV PPO 训练管道启动")
    print_info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print_warn("CUDA 不可用，将使用 CPU。建议使用 GPU 加速训练。")
    else:
        print_info(f"使用设备: {device}")

    # --- 初始化环境 ---
    env = Env(cfg)
    print_info("环境初始化完成")

    # --- 初始化 PPO 训练器 ---
    trainer = PPOTrainer(cfg, device=device)
    print_info("PPO 训练器初始化完成")

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

        # 周期性保存 checkpoint
        if (epoch + 1) % (cfg.trainer.checkpoint_interval // cfg.trainer.steps_per_epoch) == 0:
            trainer.save_checkpoint(checkpoint_dir)

    # --- 训练完成 ---
    print_info("=" * 60)
    print_info("训练完成")
    print_info("=" * 60)
    trainer.save_checkpoint(checkpoint_dir)
    print_info(f"最终检查点已保存至: {checkpoint_dir}")


if __name__ == "__main__":
    main()


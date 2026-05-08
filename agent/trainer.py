"""PPO (Proximal Policy Optimization) 训练器实现。

该模块实现 PPO 算法的训练逻辑，包括轨迹收集、策略更新与检查点管理。
"""

from typing import Tuple, Dict, Any
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np
from omegaconf import DictConfig

from agent.acnet import ACNet
from buffer.rollout_buffer import RolloutBuffer
from utils.rich_print import print_info, print_warn, print_error, print_success


class PPOTrainer:
    """PPO 训练器，用于 UUV 隐蔽突防任务的强化学习训练。

    集成 ACNet 神经网络、优化器、经验缓冲区与 PPO 算法更新流程。
    所有计算在 GPU 上执行。

    调用示例:
        >>> cfg = DictConfig({'trainer': {...}, 'ppo': {...}})
        >>> trainer = PPOTrainer(cfg, device="cuda")
        >>> 
        >>> for epoch in range(10):
        ...     trajectory_info = trainer.collect_rollout(env, num_steps=1000)
        ...     update_info = trainer.update_policy()
        ...     print(f"Epoch {epoch}: reward={trajectory_info['reward_mean']:.2f}, "
        ...           f"loss={update_info['policy_loss']:.4f}")
    """

    def __init__(self, cfg: DictConfig, device: str = "cuda", env=None):
        """初始化 PPO 训练器。

        功能说明:
            初始化 ACNet、优化器、RolloutBuffer 与 PPO 超参数。
            所有模型参数与计算直接在指定设备上。

        输入参数:
            cfg (DictConfig): Hydra 配置对象，包含 trainer 与 ppo 配置段。
            device (str): 计算设备，默认为 "cuda"。
            env: 环境对象，用于 RolloutBuffer 动态重构 spatial_input。

        输出参数:
            无。

        调用示例:
            >>> trainer = PPOTrainer(cfg, device="cuda", env=env)
        """
        self.cfg = cfg
        self.device = device
        self.env = env

        # --- 初始化 ACNet 网络 ---
        self.model = ACNet(device=device).to(device)
        print_info(f"ACNet 初始化完成，设备: {device}")

        # --- 初始化优化器（添加权重衰减 L2 正则化）---
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg.ppo.learning_rate,
            weight_decay=1e-4,
        )
        print_info(f"优化器初始化完成，学习率: {cfg.ppo.learning_rate}，权重衰减: 1e-4")

        # --- 初始化 RolloutBuffer（传入 env 用于动态重构 spatial_input）---
        self.buffer = RolloutBuffer(
            capacity=cfg.ppo.rollout_steps,
            device=device,
            env=env,
        )
        print_info(f"RolloutBuffer 初始化完成，容量: {cfg.ppo.rollout_steps}，spatial_input 动态重构已启用")

        # --- 初始化学习率调度器（线性预热 + 余弦衰减）---
        max_epochs = cfg.trainer.max_epochs
        warmup_steps = max(1, cfg.ppo.rollout_steps // 10)  # 前 10% 的 rollout steps 进行预热
        self._warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_steps,
        )
        self._cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs,
            eta_min=1e-6,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[self._warmup_scheduler, self._cosine_scheduler],
            milestones=[warmup_steps],
        )
        print_info(f"LR调度器初始化完成：预热步数={warmup_steps}，最大epoch={max_epochs}")

        # --- PPO 超参数 ---
        self.gamma = cfg.ppo.gamma
        self.gae_lambda = cfg.ppo.gae_lambda
        self.clip_ratio = cfg.ppo.clip_ratio
        self.value_coef = cfg.ppo.value_coef
        self.entropy_coef = cfg.ppo.entropy_coef
        self.epochs = cfg.ppo.epochs
        self.minibatch_size = cfg.ppo.minibatch_size
        self.checkpoint_interval = cfg.trainer.checkpoint_interval

        # --- 训练状态 ---
        self.global_step = 0
        self.episode = 0
        self.current_epoch = 0

    def collect_rollout(
        self, env: Any, num_steps: int, max_episode_steps: int = 10000
    ) -> Dict[str, float]:
        """从环境中收集轨迹数据。

        功能说明:
            与环境交互，收集 num_steps 步的转移数据。
            每次调用后，RolloutBuffer 中存储了新轨迹且已清空之前的数据。

        输入参数:
            env (Any): 环境对象，具有 reset()、step()、get_observation_tensor() 方法。
            num_steps (int): 要收集的步数目标。
            max_episode_steps (int): 单个 episode 的最大步数限制，默认 10000。

        输出参数:
            Dict[str, float]:
                {
                    'reward_mean': float,  # 平均每步奖励
                    'episode_rewards': float,  # 当前 episode 累计奖励
                    'episode_length': int,  # 当前 episode 步数
                    'num_episodes': int,  # 本轮收集的 episode 数
                }

        调用示例:
            >>> info = trainer.collect_rollout(env, num_steps=1000)
            >>> print(f"收集完成，平均奖励: {info['reward_mean']:.2f}")
        """
        self.buffer.clear()
        self.model.eval()

        total_reward = 0.0
        episode_reward = 0.0
        episode_length = 0
        num_episodes = 0
        step_count = 0

        env.reset()

        with torch.no_grad():
            while step_count < num_steps:
                # 获取当前观测
                spatial_input, state_vector = env.get_observation_tensor(device=self.device)
                assert state_vector.shape == (1, 7), f"状态向量维度错误，期望 (1, 7)，实际 {state_vector.shape}"

                # 前向传播，获取动作 logits 与价值估计
                actor_logits, value = self.model(spatial_input, state_vector)

                # 采样动作
                action_probs = torch.softmax(actor_logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                logprob = action_dist.log_prob(action)

                # 执行环境步
                next_pos, reward, done, info = env.step(action.item())
                episode_reward += reward
                episode_length += 1

                # 将经验加入缓冲区
                self.buffer.add(
                    spatial_input=spatial_input,
                    state_vector=state_vector,
                    action=action,
                    logprob=logprob,
                    reward=reward,
                    done=done,
                    value=value,
                )

                total_reward += reward
                step_count += 1

                # 处理 episode 结束
                if done or episode_length >= max_episode_steps:
                    # 计算终止状态的价值估计
                    if done:
                        next_value = 0.0
                    else:
                        next_spatial, next_state_vec = env.get_observation_tensor(
                            device=self.device
                        )
                        _, next_value = self.model(next_spatial, next_state_vec)
                        next_value = next_value.item()

                    # 对于成功到达终点的情况，日志使用 print_success 打印
                    if "胜利" in info.get('result', ''):
                        print_success(
                            f"Episode {num_episodes} 结束: 奖励={episode_reward:.2f}, "
                            f"步数={episode_length}, 理由={info.get('result', 'max_steps')}"
                        )
                    else:
                        print_info(
                            f"Episode {num_episodes} 结束: 奖励={episode_reward:.2f}, "
                            f"步数={episode_length}, 理由={info.get('result', 'max_steps')}"
                        )

                    num_episodes += 1
                    total_reward += episode_reward
                    episode_reward = 0.0
                    episode_length = 0

                    # 重置环境
                    env.reset()

        # 计算 GAE 与回报
        self.buffer.compute_gae_returns(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            next_value=0.0,
        )

        reward_mean = total_reward / step_count if step_count > 0 else 0.0

        print_info(
            f"Rollout 收集完成: 步数={step_count}, "
            f"episode={num_episodes}, 平均奖励={reward_mean:.4f}"
        )

        self.episode += num_episodes
        
        # 获取当前学习率用于日志
        current_lr = self.optimizer.param_groups[0]['lr']
        
        return {
            "reward_mean": reward_mean,
            "episode_rewards": total_reward,
            "episode_length": episode_length,
            "num_episodes": num_episodes,
            "current_lr": current_lr,
        }

    def update_policy(self) -> Dict[str, float]:
        """执行 PPO 策略更新。

        功能说明:
            使用 RolloutBuffer 中的数据进行多 epoch、多 minibatch 的 PPO 更新。
            每个 minibatch 计算策略损失（PPO-Clip）、价值函数损失与熵损失。

        输入参数:
            无。

        输出参数:
            Dict[str, float]:
                {
                    'policy_loss': float,  # 策略损失均值
                    'value_loss': float,   # 价值函数损失均值
                    'entropy': float,      # 动作熵均值
                    'total_loss': float,   # 总损失均值
                }

        调用示例:
            >>> update_info = trainer.update_policy()
            >>> print(f"策略损失: {update_info['policy_loss']:.4f}")
        """
        self.model.train()

        policy_losses = []
        value_losses = []
        entropies = []
        total_losses = []

        for epoch in range(self.epochs):
            for minibatch in self.buffer.iterate_minibatches(self.minibatch_size):
                # 提取 minibatch 数据
                spatial_input = minibatch["spatial_input"]
                state_vector = minibatch["state_vector"]
                actions = minibatch["action"]
                old_logprobs = minibatch["logprob"]
                returns = minibatch["return"]
                advantages = minibatch["advantage"]

                # 前向传播
                actor_logits, values = self.model(spatial_input, state_vector)

                # 计算新的 logprob 和熵
                action_probs = torch.softmax(actor_logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                new_logprobs = action_dist.log_prob(actions)
                entropy = action_dist.entropy().mean()

                # --- PPO-Clip 策略损失 ---
                ratio = torch.exp(new_logprobs - old_logprobs)
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- 价值函数损失 ---
                value_loss = nn.MSELoss()(values.squeeze(-1), returns)

                # --- 总损失 ---
                total_loss = (
                    policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                )

                # 反向传播与优化
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                # 记录损失
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
                total_losses.append(total_loss.item())

                self.global_step += 1

        avg_policy_loss = np.mean(policy_losses) if policy_losses else 0.0
        avg_value_loss = np.mean(value_losses) if value_losses else 0.0
        avg_entropy = np.mean(entropies) if entropies else 0.0
        avg_total_loss = np.mean(total_losses) if total_losses else 0.0

        print_info(
            f"PPO 更新完成 (global_step={self.global_step}): "
            f"策略损失={avg_policy_loss:.4f}, 价值损失={avg_value_loss:.4f}, "
            f"熵={avg_entropy:.4f}, 总损失={avg_total_loss:.4f}"
        )

        # 更新学习率调度器
        self.scheduler.step()
        self.current_epoch += 1
        current_lr = self.optimizer.param_groups[0]['lr']
        print_info(f"Epoch {self.current_epoch} LR调度完成，当前学习率: {current_lr:.6f}")

        return {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
            "total_loss": avg_total_loss,
            "current_lr": current_lr,
        }

    def save_checkpoint(self, checkpoint_dir: Path) -> None:
        """保存训练检查点。

        功能说明:
            保存模型参数、优化器状态与训练元数据（全局步数、回合数）。
            输出目录应为 `outputs/<date>/<time>/checkpoints/`。

        输入参数:
            checkpoint_dir (Path): 检查点输出目录（绝对路径）。

        输出参数:
            无。在 checkpoint_dir 下生成：
                model_step_<global_step>.pt
                optimizer_step_<global_step>.pt
                train_state_step_<global_step>.npz

        调用示例:
            >>> ckpt_dir = Path("outputs/2026-04-21/10-44-17/checkpoints")
            >>> trainer.save_checkpoint(ckpt_dir)
        """
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model_path = checkpoint_dir / f"model_step_{self.global_step}.pt"
        optimizer_path = checkpoint_dir / f"optimizer_step_{self.global_step}.pt"
        state_path = checkpoint_dir / f"train_state_step_{self.global_step}.npz"

        # 保存模型与优化器
        torch.save(self.model.state_dict(), str(model_path))
        torch.save(self.optimizer.state_dict(), str(optimizer_path))

        # 保存训练元状态
        np.savez_compressed(
            str(state_path),
            global_step=self.global_step,
            episode=self.episode,
        )

        print_info(
            f"检查点已保存: "
            f"model={model_path.name}, "
            f"optimizer={optimizer_path.name}, "
            f"state={state_path.name}"
        )

    def load_checkpoint(self, checkpoint_dir: Path, step: int) -> None:
        """加载训练检查点。

        功能说明:
            从检查点恢复模型参数、优化器状态与训练进度（全局步数、回合数）。

        输入参数:
            checkpoint_dir (Path): 检查点所在目录。
            step (int): 要加载的全局步数（用于构建文件名）。

        输出参数:
            无。模型与优化器状态被加载到当前对象。

        调用示例:
            >>> trainer.load_checkpoint(ckpt_dir, step=5000)
        """
        model_path = checkpoint_dir / f"model_step_{step}.pt"
        optimizer_path = checkpoint_dir / f"optimizer_step_{step}.pt"
        state_path = checkpoint_dir / f"train_state_step_{step}.npz"

        if not model_path.exists():
            raise FileNotFoundError(f"模型检查点不存在: {model_path}")

        # 加载模型
        self.model.load_state_dict(
            torch.load(str(model_path), map_location=self.device)
        )

        # 加载优化器
        if optimizer_path.exists():
            self.optimizer.load_state_dict(
                torch.load(str(optimizer_path), map_location=self.device)
            )

        # 加载训练元状态
        if state_path.exists():
            meta = np.load(str(state_path))
            self.global_step = int(meta["global_step"])
            self.episode = int(meta["episode"])

        print_info(
            f"检查点已加载: "
            f"global_step={self.global_step}, episode={self.episode}"
        )

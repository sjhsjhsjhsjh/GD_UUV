"""On-policy Rollout Buffer 实现。

该模块提供全 GPU 的轨迹缓存与 GAE 优势计算。
"""

from typing import Tuple, Generator, Dict
import torch
import numpy as np


class RolloutBuffer:
    """On-policy 轨迹缓冲区，支持 GAE 优势计算与 minibatch 迭代。

    缓冲区预分配 CUDA 张量，存储单个 episode 或指定步数的轨迹。
    支持 Generalized Advantage Estimation (GAE) 计算与小批量迭代。

    调用示例:
        >>> buffer = RolloutBuffer(capacity=1000, device="cuda")
        >>> for step in range(episode_steps):
        ...     spatial_input, state_vector = env.get_observation_tensor(device="cuda")
        ...     actor_logits, value = model(spatial_input, state_vector)
        ...     action = torch.multinomial(torch.softmax(actor_logits, dim=-1), 1).squeeze()
        ...     logprob = torch.nn.functional.log_softmax(actor_logits, dim=-1)[0, action]
        ...     
        ...     next_pos, reward, done, info = env.step(action.item())
        ...     next_spatial, next_state_vec = env.get_observation_tensor(device="cuda")
        ...     _, next_value = model(next_spatial, next_state_vec)
        ...     
        ...     buffer.add(spatial_input, state_vector, action, logprob, reward, done, value, next_value)
        ...     
        ...     if done:
        ...         break
        >>> 
        >>> buffer.compute_gae_returns(gamma=0.99, gae_lambda=0.95)
        >>> for minibatch in buffer.iterate_minibatches(batch_size=32):
        ...     # 执行 PPO 更新
        ...     pass
    """

    def __init__(self, capacity: int = 10000, device: str = "cuda", env=None):
        """初始化缓冲区。

        功能说明:
            预分配 CUDA 张量存储轨迹数据。spatial_input 将在 iterate_minibatches 时动态重构。

        输入参数:
            capacity (int): 缓冲区容量（最多存储的步数），默认 10000。
            device (str): 计算设备，默认为 "cuda"。
            env: 环境对象，用于动态重构 spatial_input。

        输出参数:
            无。

        调用示例:
            >>> buffer = RolloutBuffer(capacity=5000, device="cuda", env=env)
        """
        self.capacity = capacity
        self.device = device
        self.env = env
        self.ptr = 0  # 当前写入指针

        # --- 预分配张量存储轨迹 ---
        # spatial_input: 不再存储，将在 iterate_minibatches 时动态重构
        self._state_vector = None
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.logprobs = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.values = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.returns = None  # 在 compute_gae_returns 时计算
        self.advantages = None  # 在 compute_gae_returns 时计算

    def add(
        self,
        spatial_input: torch.Tensor,
        state_vector: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: float,
        done: bool,
        value: torch.Tensor,
    ) -> None:
        """添加一步经验到缓冲区。

        功能说明:
            将单步经验（观测、动作、奖励、价值等）存储到预分配的张量中。
            spatial_input 张量已弃用（不再存储），将在 iterate_minibatches 时动态重构。
            state_vector 张量形状应为 (1, 7) 的 batch 形式。

        输入参数:
            spatial_input (torch.Tensor): 已弃用，可传入 None。原为形状 (1, 2, D, H, W)。
            state_vector (torch.Tensor): 形状 (1, 7)，状态向量。
            action (torch.Tensor): 形状 () 或 (1,)，执行的动作编号。
            logprob (torch.Tensor): 形状 () 或 (1,)，动作的 log 概率。
            reward (float): 获得的奖励值。
            done (bool): 是否游戏结束。
            value (torch.Tensor): 形状 () 或 (1,)，当前状态价值估计。

        输出参数:
            无。缓冲区内部指针 self.ptr 递增。

        调用示例:
            >>> buffer.add(
            ...     spatial_input=None,  # 已弃用
            ...     state_vector=(1,7),
            ...     action=2,
            ...     logprob=-0.5,
            ...     reward=1.5,
            ...     done=False,
            ...     value=0.8
            ... )
        """
        import warnings
        
        assert self.ptr < self.capacity, f"缓冲区已满 (ptr={self.ptr}, capacity={self.capacity})"

        # spatial_input 参数已弃用
        if spatial_input is not None:
            warnings.warn(
                "spatial_input 参数已弃用。从 v2.0 开始，spatial_input 不再存储，"
                "将在 iterate_minibatches() 时根据 state_vector 动态重构。"
                "请为 spatial_input 参数传入 None 以避免此警告。",
                DeprecationWarning,
                stacklevel=2
            )

        # 第一次 add 时初始化 state_vector 张量
        if self._state_vector is None:
            state_shape = state_vector.shape
            self._state_vector = torch.zeros(
                (self.capacity,) + state_shape[1:],
                dtype=torch.float32,
                device=self.device,
            )

        # 存储状态向量（squeeze 去掉 batch 维，因为缓冲区已包含 capacity 维）
        self._state_vector[self.ptr] = state_vector.squeeze(0)

        # 存储动作、对数概率、奖励、完成标志、价值
        self.actions[self.ptr] = (
            action.item() if isinstance(action, torch.Tensor) else action
        )
        self.logprobs[self.ptr] = (
            logprob.item() if isinstance(logprob, torch.Tensor) else logprob
        )
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value.item() if isinstance(value, torch.Tensor) else value

        self.ptr += 1

    def _reconstruct_spatial_input_batch(self, batch_state_vector: torch.Tensor) -> torch.Tensor:
        """根据 batch state_vector 动态重构 spatial_input（批量版本）。

        功能说明:
            从 state_vector 中提取 UUV 和敌人坐标，利用环境中的地图数据生成观测张量。
            所有计算在 GPU 上执行，支持批处理。

        输入参数:
            batch_state_vector (torch.Tensor): 形状 (batch_size, 7)，包含坐标信息
                - [:, 0]: x_uuv
                - [:, 1]: y_uuv
                - [:, 2]: z_uuv
                - [:, 3]: y_enemy
                - [:, 4-6]: 其他状态（不用于重构）

        输出参数:
            torch.Tensor: 形状 (batch_size, 2, D, H, W)，重构的 spatial_input 张量

        调用示例:
            >>> batch_state_vec = torch.randn(32, 7, device="cuda")
            >>> spatial_batch = buffer._reconstruct_spatial_input_batch(batch_state_vec)
            >>> spatial_batch.shape
            torch.Size([32, 2, 16, 16, 11])
        """
        import torch
        
        assert self.env is not None, "重构 spatial_input 需要环境对象 (env)。请在 __init__ 时传入 env 参数。"
        
        batch_size = batch_state_vector.shape[0]
        # 从环境中获取观测参数
        field_of_view = self.env.field_of_view
        field_of_view_on_z = self.env.field_of_view_on_z
        
        # 预分配输出张量
        spatial_batch = torch.zeros(
            (batch_size, 2, field_of_view, field_of_view, field_of_view_on_z),
            dtype=torch.float32,
            device=self.device
        )
        
        # 逐样本重构 spatial_input
        for i in range(batch_size):
            # 提取坐标（转换为整数）
            x_uuv = int(batch_state_vector[i, 0].item())
            y_uuv = int(batch_state_vector[i, 1].item())
            z_uuv = int(batch_state_vector[i, 2].item())
            y_enemy = int(batch_state_vector[i, 3].item())
            
            # 临时设置环境中的机器人位置（单线程环境中安全）
            original_uuv_x = self.env.uuv.x
            original_uuv_y = self.env.uuv.y
            original_uuv_z = self.env.uuv.z
            original_enemy_y = self.env.enemy.y
            
            self.env.uuv.x = x_uuv
            self.env.uuv.y = y_uuv
            self.env.uuv.z = z_uuv
            self.env.enemy.y = y_enemy
            
            # 调用环境的 get_observation_tensor 获取观测
            obs_spatial, _ = self.env.get_observation_tensor(device=self.device)
            # obs_spatial 形状为 (1, 2, D, H, W)，需要 squeeze 第一维并存入 batch
            spatial_batch[i] = obs_spatial.squeeze(0)
            
            # 恢复环境状态
            self.env.uuv.x = original_uuv_x
            self.env.uuv.y = original_uuv_y
            self.env.uuv.z = original_uuv_z
            self.env.enemy.y = original_enemy_y
        
        return spatial_batch

    def compute_gae_returns(
        self, gamma: float = 0.99, gae_lambda: float = 0.95, next_value: float = 0.0
    ) -> None:
        """计算 Generalized Advantage Estimation (GAE) 与回报。

        功能说明:
            使用 TD 残差（Temporal Difference Residual）递推计算 GAE 优势估计。
            公式：
                td_residual[t] = reward[t] + gamma * (1 - done[t]) * value[t+1] - value[t]
                advantages[t] = td_residual[t] + (gamma * gae_lambda) * (1 - done[t]) * advantages[t+1]
                returns[t] = advantages[t] + value[t]

        输入参数:
            gamma (float): 折扣因子，默认 0.99。
            gae_lambda (float): GAE 衰减系数，默认 0.95。
            next_value (float): Episode 结束后的价值估计（通常为 0），默认 0.0。

        输出参数:
            无。计算结果存储到 self.returns 与 self.advantages。

        调用示例:
            >>> buffer.compute_gae_returns(gamma=0.99, gae_lambda=0.95, next_value=0.0)
        """
        num_steps = self.ptr
        self.advantages = torch.zeros(num_steps, dtype=torch.float32, device=self.device)
        self.returns = torch.zeros(num_steps, dtype=torch.float32, device=self.device)

        # 反向递推计算 GAE
        last_value = next_value
        last_advantage = 0.0

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_value_t = next_value
            else:
                next_value_t = self.values[t + 1].item()

            # 计算 TD 残差
            td_residual = (
                self.rewards[t]
                + gamma * (1 - self.dones[t].float()) * next_value_t
                - self.values[t].item()
            )

            # 累积 GAE 优势
            advantage = (
                td_residual
                + gamma * gae_lambda * (1 - self.dones[t].float()) * last_advantage
            )

            self.advantages[t] = advantage
            self.returns[t] = advantage + self.values[t]

            last_advantage = advantage

    def iterate_minibatches(self, batch_size: int) -> Generator[Dict, None, None]:
        """生成经过随机打乱的 minibatch。

        功能说明:
            将缓冲区中的数据随机分割成指定大小的 minibatch。
            spatial_input 将根据 state_vector 在此函数中动态重构。

        输入参数:
            batch_size (int): 每个 minibatch 的大小。

        输出参数:
            Generator：每次迭代返回一个包含如下字段的字典：
                {
                    'spatial_input': (batch_size, 2, D, H, W),  # 动态重构
                    'state_vector': (batch_size, 7),
                    'action': (batch_size,),
                    'logprob': (batch_size,),
                    'return': (batch_size,),
                    'advantage': (batch_size,),
                    'value': (batch_size,),
                }

        调用示例:
            >>> for minibatch in buffer.iterate_minibatches(batch_size=32):
            ...     spatial_input = minibatch['spatial_input']
            ...     state_vector = minibatch['state_vector']
            ...     actions = minibatch['action']
            ...     logprobs = minibatch['logprob']
            ...     returns = minibatch['return']
            ...     advantages = minibatch['advantage']
            ...     # ... 执行 PPO 更新
        """
        assert self.returns is not None, "必须先调用 compute_gae_returns()"
        assert self.advantages is not None, "必须先调用 compute_gae_returns()"

        num_steps = self.ptr
        indices = torch.randperm(num_steps, device=self.device)

        for start_idx in range(0, num_steps, batch_size):
            end_idx = min(start_idx + batch_size, num_steps)
            batch_indices = indices[start_idx:end_idx]
            
            # 获取 batch state_vector
            batch_state_vector = self._state_vector[batch_indices]
            
            # 动态重构 spatial_input
            batch_spatial_input = self._reconstruct_spatial_input_batch(batch_state_vector)

            minibatch = {
                "spatial_input": batch_spatial_input,
                "state_vector": batch_state_vector,
                "action": self.actions[batch_indices],
                "logprob": self.logprobs[batch_indices],
                "return": self.returns[batch_indices],
                "advantage": self.advantages[batch_indices],
                "value": self.values[batch_indices],
            }
            yield minibatch

    def clear(self) -> None:
        """清空缓冲区，重置写入指针。

        功能说明:
            重置缓冲区状态以支持下一个 episode 或 rollout。

        输入参数:
            无。

        输出参数:
            无。

        调用示例:
            >>> buffer.clear()
        """
        self.ptr = 0
        self.returns = None
        self.advantages = None

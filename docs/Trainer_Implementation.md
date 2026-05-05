# ACNet Trainer + GPU Buffer 实现文档

**文档版本**: 1.0  
**创建日期**: 2026-04-27  
**实现状态**: ✅ 完成  
**冒烟测试**: ✅ 6/6 通过

---

## 1. 项目背景与目标

### 1.1 背景

本项目基于 `ACNet`（Actor-Critic 神经网络）为 UUV（无人水下航行器）隐蔽突防任务设计了强化学习训练系统。在此之前，项目仅有网络架构定义，缺少：

- **训练器**: 完整的强化学习算法实现（策略收集、梯度更新）
- **缓冲区**: 全 GPU 的经验存储与批处理机制
- **一键入口**: 可独立运行的训练脚本

### 1.2 实现目标

设计并实现以下模块，使得整个系统能在 GPU 上无缝运行，无需命令行参数即可一键启动训练：

1. ✅ **环境观测接口** - 直接返回 ACNet 可用的张量格式
2. ✅ **On-policy Rollout Buffer** - 全 GPU 轨迹缓存与 GAE 优势计算
3. ✅ **PPO Trainer** - 完整的 PPO 策略优化器与 checkpoint 管理
4. ✅ **配置系统** - 统一的超参数定义与管理
5. ✅ **训练入口** - Hydra 驱动的一键启动脚本

---

## 2. 实现架构与分阶段设计

### 2.1 五阶段分解

| 阶段 | 名称 | 输出 | 状态 |
|------|------|------|------|
| A | 环境观测接口 | `env.get_observation_tensor()` | ✅ 完成 |
| B | GPU Rollout Buffer | `RolloutBuffer` 类 | ✅ 完成 |
| C | PPO Trainer 核心 | `PPOTrainer` 类 | ✅ 完成 |
| D | 配置与入口 | `main_config.yaml` + `runs.py` | ✅ 完成 |
| E | 冒烟验证 | `smoke_test.py` | ✅ 6/6 通过 |

### 2.2 模块依赖关系

```
runs.py (训练入口)
  ├── Env (环境)
  │   └── get_observation_tensor() [A阶段]
  └── PPOTrainer [C阶段]
      ├── ACNet (模型，复用)
      └── RolloutBuffer [B阶段]
          ├── Env.get_observation_tensor()
          └── ACNet
```

---

## 3. 核心模块详解

### 3.1 阶段 A: 环境观测接口

**文件**: `env/env.py::Env.get_observation_tensor()`

**功能**: 将环境状态转换为 ACNet 可直接使用的张量对

**方法签名**
```python
def get_observation_tensor(self, device: str = "cuda", window_size: int = 16) -> Tuple[torch.Tensor, torch.Tensor]
```

**返回值**
- `spatial_input`: 形状 `(1, 2, window_size, window_size, window_size)`, dtype=float32, device=cuda
  - 通道 0: 传播损失 (TL) 热力图 - 围绕 UUV 位置的局部窗口内各点到敌方的 TL 值
  - 通道 1: 地形可通行性 - 1.0=不可通行(True), 0.0=可通行(False)
- `state_vector`: 形状 `(1, 6)`, dtype=float32, device=cuda
  - 内容: `[x_uuv, y_uuv, z_uuv, x_enemy, y_enemy, z_enemy]`（网格索引，非米数）

**坐标系说明**
- 所有坐标存储在 `Robot` 对象中为**网格索引**（0-100 范围）
- 配置文件中的坐标为**米数**，需在 `reset()` 中转换：
  - X/Y: 配置米数 ÷ 100 = 网格索引（采样步长 100m）
  - Z: 配置米数 ÷ 50 = 网格索引（采样步长 50m）

**关键实现细节**
1. **窗口映射**: 计算窗口中心（UUV 位置）周围 `window_size × window_size × window_size` 的立方体
2. **边界处理**: 当窗口超出地形边界时，只填充可访问的区域，其余置 0
3. **张量设备**: 所有张量在方法内直接在 CUDA 上创建，无需事后转移

**调用示例**
```python
env = Env(cfg)
env.reset()
spatial_input, state_vector = env.get_observation_tensor(device="cuda", window_size=16)
print(spatial_input.shape)   # torch.Size([1, 2, 16, 16, 16])
print(state_vector.shape)    # torch.Size([1, 6])
print(spatial_input.device)  # cuda:0
```

---

### 3.2 阶段 B: GPU On-policy Rollout Buffer

**文件**: `buffer/rollout_buffer.py::RolloutBuffer`

**用途**: 存储单个 episode 或固定步数的轨迹，支持 Generalized Advantage Estimation (GAE) 计算与 minibatch 迭代

**核心方法**

#### `__init__(capacity: int, device: str)`
预分配 CUDA 张量，初始化缓冲区状态

```python
buffer = RolloutBuffer(capacity=2000, device="cuda")
```

#### `add(spatial_input, state_vector, action, logprob, reward, done, value)`
添加单步经验，缓冲区内部自动初始化 spatial/state 张量（首次调用时）

```python
buffer.add(
    spatial_input=torch.randn(1, 2, 16, 16, 16, device="cuda"),
    state_vector=torch.randn(1, 6, device="cuda"),
    action=torch.tensor([2], device="cuda"),
    logprob=torch.tensor(-0.5, device="cuda"),
    reward=1.5,
    done=False,
    value=torch.tensor(0.8, device="cuda")
)
```

#### `compute_gae_returns(gamma, gae_lambda, next_value)`
计算 GAE 优势与回报（episode 结束时调用）

```python
buffer.compute_gae_returns(gamma=0.99, gae_lambda=0.95, next_value=0.0)
```

**计算公式**
$$\text{TD\_residual}[t] = r_t + \gamma(1-d_t)V(s_{t+1}) - V(s_t)$$
$$\text{advantages}[t] = \text{TD\_residual}[t] + \gamma\lambda(1-d_t)\text{advantages}[t+1]$$
$$\text{returns}[t] = \text{advantages}[t] + V(s_t)$$

#### `iterate_minibatches(batch_size)`
生成随机打乱的 minibatch，用于 PPO 多 epoch 更新

```python
for minibatch in buffer.iterate_minibatches(batch_size=64):
    spatial = minibatch['spatial_input']        # (batch_size, 2, 16, 16, 16)
    state_vec = minibatch['state_vector']       # (batch_size, 6)
    actions = minibatch['action']               # (batch_size,)
    old_logprobs = minibatch['logprob']         # (batch_size,)
    returns = minibatch['return']               # (batch_size,)
    advantages = minibatch['advantage']         # (batch_size,)
    # ... 执行 PPO 更新
```

#### `clear()`
重置缓冲区指针，清空返回值与优势估计

```python
buffer.clear()
```

**数据存储结构**
- 预分配张量（所有在 CUDA）:
  - `_spatial_input`: (capacity, 2, D, H, W)
  - `_state_vector`: (capacity, 6)
  - `actions`: (capacity,), dtype=int64
  - `logprobs`: (capacity,), dtype=float32
  - `rewards`: (capacity,), dtype=float32
  - `dones`: (capacity,), dtype=bool
  - `values`: (capacity,), dtype=float32
- 计算得出（调用 GAE 后）:
  - `returns`: (capacity,), dtype=float32
  - `advantages`: (capacity,), dtype=float32

---

### 3.3 阶段 C: PPO Trainer 核心

**文件**: `agent/trainer.py::PPOTrainer`

**用途**: 完整的 PPO 强化学习训练器，集成模型、优化器、缓冲区、checkpoint 管理

**核心方法**

#### `__init__(cfg: DictConfig, device: str = "cuda")`
初始化 ACNet、优化器、RolloutBuffer 与 PPO 超参数

```python
from omegaconf import DictConfig
cfg = DictConfig({
    "ppo": {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        ...
    },
    "trainer": {
        "checkpoint_interval": 1000,
        ...
    }
})
trainer = PPOTrainer(cfg, device="cuda")
```

#### `collect_rollout(env, num_steps, max_episode_steps) -> dict`
与环境交互，收集 `num_steps` 步的轨迹数据，返回统计信息

**工作流程**
1. 重置环境
2. 循环执行 `num_steps` 步：
   - 调用 `env.get_observation_tensor()` 获取观测
   - 通过 ACNet forward 获取动作 logits 与价值估计
   - 采样动作（使用 Categorical 分布）
   - 执行 `env.step(action)`
   - 将经验加入 RolloutBuffer
3. Episode 结束时（done=True 或 max_steps），计算下一状态价值并重置环境
4. 最后调用 `buffer.compute_gae_returns()` 计算 GAE

**返回值**
```python
{
    'reward_mean': float,          # 平均每步奖励
    'episode_rewards': float,      # 总累计奖励
    'episode_length': int,         # 最后一个 episode 的步数
    'num_episodes': int            # 本轮收集的 episode 数
}
```

#### `update_policy() -> dict`
执行 PPO 策略更新（多 epoch, 多 minibatch）

**工作流程**
1. 对于每个 epoch（默认 3 次）：
   - 遍历所有 minibatch
   - 计算新的 action logprob 与动作熵
   - **PPO-Clip 策略损失**:
     $$L^{clip} = -\mathbb{E}[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$
   - **价值函数损失**: $L^V = \text{MSE}(\hat{V}(s_t) - R_t)$
   - **总损失**: $L = L^{clip} + c_1 L^V - c_2 H[\pi]$（$c_1$=value_coef, $c_2$=entropy_coef）
   - 反向传播与梯度裁剪

**返回值**
```python
{
    'policy_loss': float,
    'value_loss': float,
    'entropy': float,
    'total_loss': float
}
```

#### `save_checkpoint(checkpoint_dir: Path) -> None`
保存模型检查点（仅模型与优化器，不保存 buffer 数据）

**输出文件**
- `model_step_{global_step}.pt` - 模型参数字典
- `optimizer_step_{global_step}.pt` - 优化器状态字典
- `train_state_step_{global_step}.npz` - 训练元数据（global_step, episode）

#### `load_checkpoint(checkpoint_dir: Path, step: int) -> None`
从检查点恢复训练状态

```python
trainer.load_checkpoint(Path("outputs/2026-04-21/10-44-17/checkpoints"), step=5000)
# 模型、优化器、全局步数都会恢复
print(f"恢复后 global_step: {trainer.global_step}")
```

**PPO 超参数说明**
| 参数 | 默认值 | 说明 |
|------|-------|------|
| `learning_rate` | 3e-4 | Adam 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE 衰减系数 |
| `clip_ratio` | 0.2 | PPO 裁剪范围 $\epsilon$ |
| `value_coef` | 0.5 | 价值损失权重 $c_1$ |
| `entropy_coef` | 0.01 | 熵损失权重 $c_2$ |
| `epochs` | 3 | 每个 rollout 的策略更新 epoch 数 |
| `minibatch_size` | 64 | Minibatch 大小 |
| `rollout_steps` | 2000 | 缓冲区容量（等于 steps_per_epoch） |

---

### 3.4 阶段 D: 配置与训练入口

#### 配置文件: `configs/main_config.yaml`

**新增 trainer 与 ppo 配置段**

```yaml
trainer:
  checkpoint_interval: 1000          # 检查点保存间隔（步数）
  max_epochs: 100                    # 最大训练回合数
  steps_per_epoch: 2000              # 每个 epoch 的 rollout 步数

ppo:
  learning_rate: 3e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_ratio: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  epochs: 3                          # PPO 更新 epoch
  minibatch_size: 64
  rollout_steps: 2000                # 与 steps_per_epoch 保持一致
```

#### 训练入口: `runs.py`

**工作流程**
1. 使用 Hydra 加载配置（自动管理输出目录）
2. 初始化 Env 与 PPOTrainer
3. 执行训练循环：
   - 每个 epoch: `collect_rollout()` → `update_policy()`
   - 每 N 个 epoch 保存一次 checkpoint
4. 输出目录自动创建为 `outputs/<YYYY-MM-DD>/<HH-MM-SS>/`

**启动命令**
```bash
# 使用项目指定 Python 解释器
E:/lib/conda-env/torch_gpu/python.exe runs.py

# 或覆盖配置参数
E:/lib/conda-env/torch_gpu/python.exe runs.py trainer.max_epochs=50 ppo.learning_rate=5e-4
```

**日志输出示例**
```
[14:23:45] [INFO]: ============================================================
[14:23:45] [INFO]: GD-UUV PPO 训练管道启动
[14:23:45] [INFO]: ============================================================
[14:23:46] [INFO]: 使用设备: cuda
[14:23:46] [INFO]: 环境初始化完成: 地图尺寸 (101, 101, 11)
[14:23:46] [INFO]: ACNet 初始化完成，设备: cuda
[14:23:47] [INFO]: 
[14:23:47] [INFO]: --- Epoch 1/100 ---
[14:23:58] [INFO]: Episode 0 结束: 奖励=15.34, 步数=128, 理由=成功突防, 累计声信号强度: 8.23 dB
[14:24:15] [INFO]: Rollout 收集完成: 步数=2000, episode=15, 平均奖励=0.5234
[14:24:45] [INFO]: PPO 更新完成 (global_step=2000): 策略损失=0.1234, 价值损失=0.0856, 熵=1.3456, 总损失=0.1789
```

---

### 3.5 阶段 E: 冒烟验证

**文件**: `smoke_test.py`

**六项测试覆盖**
1. ✅ **导入检查** - 所有模块能否正常 import
2. ✅ **设备检查** - CUDA 是否可用
3. ✅ **ACNet 初始化** - 网络前向传播是否正常
4. ✅ **RolloutBuffer 操作** - add、GAE 计算、minibatch 迭代是否正常
5. ✅ **环境观测生成** - 环境能否生成正确格式的张量
6. ✅ **PPOTrainer 初始化** - 训练器能否初始化

**运行命令**
```bash
E:/lib/conda-env/torch_gpu/python.exe smoke_test.py
```

**测试结果示例**（全部通过）
```
[14:25:12] [INFO]: ============================================================
[14:25:12] [INFO]: GD-UUV Trainer 冒烟测试开始
[14:25:12] [INFO]: ============================================================

[14:25:12] [INFO]: ============================================================
[14:25:12] [INFO]: 测试 1: 导入检查
[14:25:12] [INFO]: ============================================================
[14:25:12] [INFO]: ✓ ACNet 导入成功
...
[14:25:25] [INFO]: ============================================================
[14:25:25] [INFO]: 测试总结
[14:25:25] [INFO]: ============================================================
[14:25:25] [INFO]: ✓ PASS: 导入检查
[14:25:25] [INFO]: ✓ PASS: 设备检查
[14:25:25] [INFO]: ✓ PASS: ACNet 初始化与推理
[14:25:25] [INFO]: ✓ PASS: RolloutBuffer 操作
[14:25:25] [INFO]: ✓ PASS: 环境观测生成
[14:25:25] [INFO]: ✓ PASS: PPOTrainer 初始化

[14:25:25] [INFO]: 总计: 6/6 通过
[14:25:25] [SUCC]: 所有冒烟测试通过！整个训练管道可用。
```

---

## 4. 关键设计决策

### 4.1 数据流与设备管理

**设计原则**: 所有张量在 GPU 上创建与计算，避免 CPU 与 GPU 间的频繁数据传输

**具体体现**
- `get_observation_tensor()` 直接在 CUDA 上创建张量（`.to(device)`）
- RolloutBuffer 预分配 CUDA 张量，所有 add 操作无设备转移开销
- PPOTrainer 的所有前向/反向都在 GPU（ACNet GPU-only 约束）

### 4.2 坐标系统与单位转换

**问题**: 配置文件使用米数（uuv_start_x_min=8000），但地形网格只有 101×101×11 个单元

**解决方案**: 在 `env.reset()` 中添加转换逻辑
```python
# 配置: 8000 米 → 网格索引: 8000 // 100 = 80
uuv_start_x_min_idx = int(self.cfg.env.uuv_start_x_min // 100)
```

**采样步长**
- X/Y: 100m（配置中 sampling_x_step/y_step）
- Z: 50m（配置中 sampling_z_step）

### 4.3 On-policy vs Replay Buffer

**选择**: On-policy Rollout Buffer

**理由**
- PPO 是 on-policy 算法，使用新采集的轨迹进行策略更新
- 无需长期存储历史数据，内存效率高
- 避免 off-policy 算法中的重要性采样偏差

**特性**
- 每个 rollout 后清空缓冲区
- 支持多 epoch 重复使用同一批数据（每 epoch 随机打乱）

### 4.4 Checkpoint 策略

**设计**: 仅保存模型与优化器，不保存 buffer

**理由**
- 避免存储大量张量（buffer 每个 rollout 都会改变）
- 训练中断后可从任意 checkpoint 继续，无需精确恢复 buffer 状态
- 显著降低磁盘 I/O 与存储占用

**checkpoint 内容**
```
checkpoints/
├── model_step_2000.pt          # 模型参数 state_dict
├── optimizer_step_2000.pt      # 优化器状态 state_dict
├── train_state_step_2000.npz   # {global_step, episode}
└── ...
```

---

## 5. 坐标系与数据格式规范

### 5.1 Robot 坐标系

**定义**: Robot 对象中的 `x, y, z` 为**网格索引**（整数）

**范围**
- X: 0-100（对应 0-10000 米）
- Y: 0-100（对应 0-10000 米）
- Z: 0-10（对应 50-550 米）

### 5.2 Spatial Input 张量格式

**形状**: `(B, 2, D, H, W)` = `(batch, channels, depth, height, width)`

**通道定义**
- 通道 0: 传播损失 (TL) 热力图
  - 单位: dB
  - 计算公式: $TL = 20 \log_{10}(distance + 1e^{-6})$
  - 值域: 通常 0~100 dB
- 通道 1: 地形可通行性
  - 1.0: 不可通行（terrain_3d[y,x,z] == True）
  - 0.0: 可通行

**索引约定**: `terrain_3d[y, x, z]`（重要！）

### 5.3 State Vector 格式

**形状**: `(B, 6)` = `(batch, state_dim)`

**内容**: `[x_uuv, y_uuv, z_uuv, x_enemy, y_enemy, z_enemy]`

**单位**: 网格索引（float32，便于神经网络处理）

---

## 6. 使用示例与工作流

### 6.1 快速启动训练

```bash
# 终端切换到项目根目录
cd e:/program/GD-UUV-self

# 使用项目配置的 Python 解释器
E:/lib/conda-env/torch_gpu/python.exe runs.py

# 或在 VS Code 中直接运行 runs.py
```

### 6.2 自定义超参数

```bash
# 覆盖默认配置参数
E:/lib/conda-env/torch_gpu/python.exe runs.py \
    trainer.max_epochs=50 \
    ppo.learning_rate=5e-4 \
    ppo.gamma=0.995 \
    trainer.steps_per_epoch=1000
```

### 6.3 从检查点继续训练

```python
# 在 runs.py 中修改或创建启动脚本
from pathlib import Path
from agent.trainer import PPOTrainer

checkpoint_dir = Path("outputs/2026-04-21/10-44-17/checkpoints")
trainer = PPOTrainer(cfg, device="cuda")
trainer.load_checkpoint(checkpoint_dir, step=5000)

# 继续训练，global_step 会自动从 5000 开始累计
for epoch in range(cfg.trainer.max_epochs):
    rollout_info = trainer.collect_rollout(env, cfg.trainer.steps_per_epoch)
    update_info = trainer.update_policy()
    trainer.global_step  # 会继续增加
```

### 6.4 评估训练进度

```python
# 在训练过程中观察 Rich 日志输出
# 关键指标：
# - 平均奖励 (reward_mean) - 应该随时间逐渐上升
# - 策略损失 (policy_loss) - 应该逐渐降低
# - 价值损失 (value_loss) - 应该逐渐降低
# - 动作熵 (entropy) - 应该逐渐降低（策略变得更确定）

# 或编写脚本从 checkpoint 加载模型进行测试
model = ACNet(device="cuda").to("cuda")
model.load_state_dict(torch.load("outputs/.../model_step_10000.pt"))
model.eval()

with torch.no_grad():
    spatial_input, state_vector = env.get_observation_tensor(device="cuda")
    actor_logits, value = model(spatial_input, state_vector)
    action = torch.argmax(actor_logits, dim=-1)  # 贪心策略
    print(f"建议动作: {action.item()}")
```

---

## 7. 常见问题与排查

### 7.1 Q: "CUDA 不可用" 错误

**原因**: 项目需要 GPU，但系统未正确安装 CUDA/cuDNN 或 PyTorch GPU 版本

**解决方案**
```bash
# 验证 CUDA 可用性
python -c "import torch; print(torch.cuda.is_available())"

# 使用项目配置的 Python 解释器（已预装 PyTorch GPU）
E:/lib/conda-env/torch_gpu/python.exe smoke_test.py
```

### 7.2 Q: 索引越界 "index X is out of bounds"

**原因**: 坐标未正确转换（米 → 网格索引），或使用了错误的索引顺序

**排查**
1. 确认 `env.reset()` 中是否进行了单位转换
2. 确认 terrain_3d 访问使用了 `[y, x, z]` 顺序（不是 `[x, y, z]`）
3. 确认网格边界检查: X/Y ∈ [0,100], Z ∈ [0,10]

### 7.3 Q: "张量不在 GPU 上" 错误

**原因**: 某处张量被误转移到 CPU，或在不同设备间传输

**排查**
```python
# 检查张量设备
print(spatial_input.device)  # 应该为 cuda:0
print(state_vector.device)   # 应该为 cuda:0

# 确保所有张量创建时直接在 GPU
tensor = torch.zeros(..., device="cuda")  # ✓ 正确
tensor = torch.zeros(...)  # ✗ CPU 上创建，然后 .to("cuda")
```

### 7.4 Q: 训练速度慢

**原因**: 
1. GPU 显存不足，导致频繁的 CPU-GPU 传输
2. 窗口大小过大（spatial_input 维度太高）
3. Minibatch 大小过小（计算效率低）

**优化建议**
```yaml
# 减小窗口大小
window_size: 8  # 默认 16，改为 8

# 增大 minibatch 大小（在显存允许的范围内）
ppo:
  minibatch_size: 128  # 从 64 改为 128

# 减少每 epoch 的步数
trainer:
  steps_per_epoch: 1000  # 从 2000 改为 1000
```

### 7.5 Q: 训练收敛不良（奖励不上升）

**原因**: 通常与奖励函数设计有关，但也可能是：
1. 学习率过低/过高
2. 网络容量不足
3. 初始化不当

**排查**
```python
# 检查初始奖励分布
for episode in range(10):
    env.reset()
    total_reward = 0
    for step in range(100):
        action = random.randint(0, 5)
        _, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    print(f"Episode {episode}: reward={total_reward}")
    
# 如果随机策略的奖励分布非常广（如 -1000 到 +1000），
# 可能需要奖励归一化或重新设计
```

---

## 8. 文件清单

### 新增文件
- `agent/trainer.py` - PPOTrainer 类
- `buffer/rollout_buffer.py` - RolloutBuffer 类
- `buffer/__init__.py` - buffer 模块导出
- `smoke_test.py` - 冒烟测试脚本
- `docs/Trainer_Implementation.md` - 本文档

### 修改文件
- `env/env.py` - 新增 `get_observation_tensor()` 方法，修复 `reset()` 中的坐标转换
- `runs.py` - 完整重写为训练入口
- `configs/main_config.yaml` - 新增 trainer 与 ppo 配置段，修复坐标参数名

### 复用文件（无修改）
- `agent/acnet.py` - ACNet 网络架构
- `env/env.py` - Env 类（仅扩展，不破坏现有方法）
- `utils/rich_print.py` - 日志工具

---

## 9. 后续优化与扩展方向

### 9.1 短期优化（建议）
1. **坐标归一化** - state_vector 除以地图尺寸 (x/100, y/100, z/10)，改进特征规范化
2. **奖励缩放** - 根据实验调整 TL 权重与 approach 权重，平衡隐蔽性与接近度
3. **多 GPU 训练** - 使用 `torch.nn.DataParallel` 或 `torch.distributed` 加速

### 9.2 中期扩展（可选）
1. **全图输入** - 替换 16×16×16 窗口为全图（101×101×11），需显存优化
2. **敌方运动** - 取消注释 `_enemy_step()`，增加环境复杂度
3. **多敌对方** - 支持多个敌方单位，提升泛化性

### 9.3 长期方向（前沿）
1. **课程学习** - 逐步增加难度（如增加敌方数量、改变奖励参数）
2. **迁移学习** - 从简单环境预训练，迁移到复杂场景
3. **模型蒸馏** - 大模型教小模型，便于部署到嵌入式系统

---

## 10. 版本历史

| 版本 | 日期 | 状态 | 备注 |
|------|------|------|------|
| 1.0 | 2026-04-27 | ✅ 完成 | 初版实现，6/6 冒烟测试通过 |

---

## 附录 A: 快速参考

### 启动训练
```bash
E:/lib/conda-env/torch_gpu/python.exe runs.py
```

### 验证安装
```bash
E:/lib/conda-env/torch_gpu/python.exe smoke_test.py
```

### 关键文件路径
- 训练器: `agent/trainer.py`
- 缓冲区: `buffer/rollout_buffer.py`
- 环境: `env/env.py` (新增 `get_observation_tensor()`)
- 配置: `configs/main_config.yaml`
- 入口: `runs.py`

### 关键类与方法
- `PPOTrainer.collect_rollout()` - 轨迹收集
- `PPOTrainer.update_policy()` - 策略更新
- `PPOTrainer.save_checkpoint()` / `.load_checkpoint()` - 检查点管理
- `RolloutBuffer.compute_gae_returns()` - GAE 计算
- `Env.get_observation_tensor()` - 观测生成

### 输出目录结构
```
outputs/
└── <YYYY-MM-DD>/
    └── <HH-MM-SS>/
        ├── checkpoints/
        │   ├── model_step_2000.pt
        │   ├── optimizer_step_2000.pt
        │   ├── train_state_step_2000.npz
        │   └── ...
        ├── .hydra/
        │   └── config.yaml          # 本次训练的配置副本
        └── hydra.log                # Hydra 日志
```

---

**文档联系**: 如有疑问或发现 bug，请参考 smoke_test.py 的诊断流程，或查阅相应模块的 docstring（函数级中文注释）。


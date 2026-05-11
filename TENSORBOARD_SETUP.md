# TensorBoard 实时训练监视系统

## 概述

项目已集成 TensorBoard 实时监视功能，能够在训练过程中实时记录和可视化以下关键指标：

- **Episode 指标**：每个 episode 的累计奖励、步数
- **训练指标**：策略损失、价值函数损失、动作熵、总损失、学习率、梯度范数
- **权重和梯度分布**：周期性记录模型权重和梯度的直方图

## 配置说明

### 配置文件位置
`configs/main_config.yaml` 中的 `trainer.tensorboard` 段：

```yaml
trainer:
  tensorboard:
    # 是否启用 TensorBoard 日志记录
    enabled: true
    # 相对于 Hydra 运行目录的日志子目录
    subdir: tensorboard
    # 标量日志记录间隔（每 N 个 minibatch 写入一次，0 表示每次都写）
    log_interval_steps: 10
    # TensorBoard 缓冲区刷新间隔（秒）
    flush_interval_seconds: 30
    # 权重和梯度直方图记录间隔（每 N 个 epoch 写入一次）
    hist_interval_epochs: 5
```

### 配置参数详解

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `enabled` | 是否启用 TensorBoard | `true` / `false` |
| `subdir` | 相对于运行目录的日志子目录 | `tensorboard` |
| `log_interval_steps` | 标量日志记录频率 | 10（降低 I/O）；0（每步记录） |
| `flush_interval_seconds` | 缓冲区刷新周期 | 30（平衡实时性和性能） |
| `hist_interval_epochs` | 直方图记录间隔 | 5 或更大（降低开销） |

## 使用方法

### 1. 运行训练

使用标准的 Hydra 命令启动训练，TensorBoard 日志会自动在每次运行的输出目录下生成：

```bash
python run.py
```

Hydra 会在 `outputs/<date>/<time>/` 目录下生成运行结果，TensorBoard 日志位于：
```
outputs/<date>/<time>/tensorboard/
```

### 2. 启动 TensorBoard 服务

在训练进行中或训练完成后，使用 TensorBoard 查看日志（假设运行目录为 `outputs/2026-05-09/10-30-45`）：

```bash
# 使用绝对路径
tensorboard --logdir "E:\program\GD-UUV-self\outputs\2026-05-09\10-30-45\tensorboard" --port 6006
```

然后在浏览器中打开：
```
http://localhost:6006
```

### 3. 实时监视多个运行

如果想同时查看多个训练运行的结果（对比不同配置），可以将多个运行目录传入 TensorBoard：

```bash
tensorboard --logdir "E:\program\GD-UUV-self\outputs" --port 6006
```

此时 TensorBoard 会自动发现所有子目录的 event 文件，并按日期/时间组织。

## 记录的指标

### Episode 指标
- `episode/reward`：每个 episode 的累计奖励
- `episode/length`：每个 episode 的步数

### 训练指标（每个 minibatch）
- `train/policy_loss`：PPO 策略损失
- `train/value_loss`：价值函数 MSE 损失
- `train/entropy`：动作熵（目标：维持充分的探索）
- `train/total_loss`：加权总损失
- `train/grad_norm`：模型参数梯度的范数（上限 0.5）
- `train/lr`：当前学习率

### 权重和梯度直方图（周期性）
- `weights/<layer_name>`：各层参数的分布
- `gradients/<layer_name>`：各层梯度的分布

## 最佳实践

### 1. 监视策略损失趋势
- 如果策略损失持续上升，可能表示学习率过高或训练不稳定
- 如果策略损失平坦，可能需要增加学习率或调整 PPO 裁剪系数

### 2. 监视梯度范数
- 梯度范数过大（接近 0.5 上限）：可能需要降低学习率
- 梯度范数过小（接近 0）：学习可能停滞，考虑增加学习率或调整网络结构

### 3. 监视 Episode 奖励
- 如果奖励逐步增加：训练进展顺利
- 如果奖励波动大：可能需要调整 episode 长度限制或奖励权重

### 4. 关闭 TensorBoard 以减少开销
若训练速度受到 I/O 影响，可在 `main_config.yaml` 中禁用：
```yaml
trainer:
  tensorboard:
    enabled: false
```

## 故障排查

### 问题 1: TensorBoard 启动时无法找到日志文件
**原因**：指定的目录不存在或路径错误  
**解决**：
- 确保使用绝对路径或运行命令时在项目根目录
- 检查运行目录中是否存在 `tensorboard/` 子目录
- 查看是否在 `configs/main_config.yaml` 中启用了 TensorBoard

### 问题 2: 在浏览器中看不到曲线
**原因**：训练尚未生成足够的数据或 TensorBoard 尚未刷新  
**解决**：
- 等待至少一个 epoch 完成
- 按 F5 刷新浏览器
- 在 TensorBoard 界面点击"Reload Data"按钮

### 问题 3: 磁盘空间占用过大
**原因**：高频率标量记录和大量直方图数据  
**解决**：
- 增加 `log_interval_steps`（例如设为 50 或 100）
- 增加 `hist_interval_epochs`（例如设为 10）
- 定期清理旧的运行日志

## 依赖安装

TensorBoard 通常随 PyTorch 一起安装，但若缺失，可手动安装：

```bash
# 使用项目指定的 Python 环境
E:/lib/conda-env/torch_gpu/python.exe -m pip install tensorboard
```

## 代码集成点

TensorBoard 集成在以下文件中：

1. **[configs/main_config.yaml](configs/main_config.yaml)**
   - 定义 TensorBoard 配置参数

2. **[agent/trainer.py](agent/trainer.py)**
   - `__init__` 方法：初始化 SummaryWriter
   - `collect_rollout` 方法：记录 episode 指标
   - `update_policy` 方法：记录训练指标和权重直方图
   - `close` 方法：关闭和刷新 TensorBoard 缓冲区

3. **[runs.py](runs.py)**
   - 从 Hydra 获取运行目录，传入 `PPOTrainer`
   - 在训练完成后调用 `trainer.close()`

## 扩展建议

若需要更高级的可视化，可添加以下功能：

1. **3D 点云可视化**：记录 UUV 位置和敌方位置的分布
2. **地形热力图**：可视化传播损失（TL）分布
3. **奖励分量分解**：单独记录隐蔽奖励、接近奖励等分量
4. **模型剪影**：在首次运行时记录网络结构

这些功能可通过在 `update_policy` 或 `collect_rollout` 中调用相应的 TensorBoard API（`add_image`, `add_figure` 等）来实现。

## 更多资源

- [TensorBoard 官方文档](https://www.tensorflow.org/tensorboard)
- [PyTorch TensorBoard 集成](https://pytorch.org/docs/stable/tensorboard.html)
- [TensorBoard 标量、直方图、图像 API](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter)

# Step 日志分析和回放工具

这是一个强化学习 **Step 级日志分析和回放工具**，用于快速定位训练日志中的游戏轮数，以及回放单轮游戏并采集详细数据。

## 功能特性

### ✨ 两种运行模式

#### 1. **count-games 模式** - 快速统计游戏轮数
- 扫描日志文件，快速统计总游戏轮数
- 生成表格展示前50轮的初始状态、敌方位置、action数量等信息
- 执行速度快，适合日志文件概览

#### 2. **replay-episode 模式** - 单轮游戏回放与数据采集
- 加载指定 episode 的初始状态
- 导入环境模块，精确重现游戏过程
- 逐步执行 action 序列，采集每步的详细数据：
  - UUV 和敌方位置
  - 传播损失 (TL) 值
  - 声纳累积信号强度
  - 奖励值
  - 游戏状态（运行中/终止/截断）
- 生成轨迹表格和摘要统计

## 日志文件格式

日志文件采用两行为一轮游戏的格式：

```
第1行（初始状态）: uuv_x,uuv_y,uuv_z,enemy_y,enemy_forward_direction
第2行（action序列）: 0-6的数字序列，以#结尾表示游戏结束

示例：
80,29,5,60,1
00142342026520330201002036444256651346040544#
87,62,4,72,1
3615365#
```

其中：
- `uuv_x, uuv_y, uuv_z`：UUV 初始位置（网格索引）
- `enemy_y`：敌方初始 Y 坐标（网格索引）
- `enemy_forward_direction`：敌方初始移动方向（1=正向, -1=反向）
- Action 编码（0-6）：
  - 0: 向前 (X-1)
  - 1: 向后 (X+1)
  - 2: 向左 (Y-1)
  - 3: 向右 (Y+1)
  - 4: 向下 (Z-1)
  - 5: 向上 (Z+1)
  - 6: 原地不动

## 使用方法

### 系统要求

- Python 3.8+
- 依赖库：`numpy`, `omegaconf`, `rich`
- 项目配置文件：`configs/main_config.yaml`
- 环境模块：`env/env.py`, `env/Robot.py`

### 安装依赖

```bash
# 依赖应该已通过项目主程序安装
# 如需单独安装
pip install omegaconf numpy rich
```

### 命令行用法

#### 模式1：统计游戏轮数

```bash
# 基本用法
python step_log_analyzer.py count-games <日志文件路径>

# 示例
python step_log_analyzer.py count-games ./outputs/2026-05-13/19-08-10/step_log.txt

# 预期输出
# - 总游戏轮数
# - 表格显示前50轮的初始状态、敌方位置、action数量等
# - 超过50轮的提示信息
```

#### 模式2：回放单轮游戏

```bash
# 基本用法
python step_log_analyzer.py replay-episode <日志文件路径> <episode索引> [选项]

# 示例（回放第5轮游戏）
python step_log_analyzer.py replay-episode ./outputs/2026-05-13/19-08-10/step_log.txt 5

# 自定义配置文件
python step_log_analyzer.py replay-episode ./outputs/2026-05-13/19-08-10/step_log.txt 0 \
    --config configs/final_tl_dashboard.yaml

# 自定义输出目录
python step_log_analyzer.py replay-episode ./outputs/2026-05-13/19-08-10/step_log.txt 0 \
    --output-dir ./custom_output
```

**可选参数：**
- `--config` (默认：`configs/main_config.yaml`)：配置文件路径
- `--output-dir` (默认：`./output`)：输出目录路径

#### 获取帮助

```bash
# 查看所有模式的帮助信息
python step_log_analyzer.py -h

# 查看特定模式的帮助信息
python step_log_analyzer.py count-games -h
python step_log_analyzer.py replay-episode -h
```

## 使用场景

### 场景1：快速了解日志文件内容
```bash
python step_log_analyzer.py count-games ./outputs/2026-05-13/19-08-10/step_log.txt
```
输出：
- 统计总游戏轮数
- 展示前50轮的初始位置、初始敌方位置、每轮 action 数量
- 快速判断日志数据的规模和质量

### 场景2：调查特定游戏的详细数据
```bash
# 查看第42轮游戏的详细轨迹
python step_log_analyzer.py replay-episode ./outputs/2026-05-13/19-08-10/step_log.txt 42
```
输出：
- 第42轮游戏的初始状态
- 逐步轨迹表格（位置、TL、奖励等）
- 游戏摘要统计（总步数、总奖励、最终结果等）

### 场景3：调试智能体行为
- 选择特定的失败/成功 episode
- 查看完整轨迹数据
- 分析 TL 变化和奖励分布
- 理解智能体的决策过程

### 场景4：数据分析和可视化
- 导出轨迹数据用于后续分析
- 查看不同阶段的游戏成功率
- 比较不同配置下的性能差异

## 输出说明

### count-games 模式输出

**表格列说明：**
| 列名 | 说明 |
|------|------|
| 轮号 | Episode 索引（0-based） |
| UUV初始位置 | 格式为 (x, y, z)，单位为网格索引 |
| 敌方Y位置 | 敌方初始 Y 坐标 |
| 敌方方向 | 正向(+1) 或 反向(-1) |
| Action数量 | 该轮游戏的 action 步数 |

### replay-episode 模式输出

**回放信息：**
- 初始状态信息
- 环境初始化状态
- 位置设置结果

**轨迹表格列说明：**
| 列名 | 说明 |
|------|------|
| 步数 | 游戏内步数索引（0-based） |
| Action | 当前步执行的动作名称 |
| UUV位置 | 当前 UUV 的 (x, y, z) 坐标 |
| Enemy位置 | 当前敌方的 (x, y, z) 坐标 |
| TL(dB) | 传播损失，单位分贝 |
| 声信号 | 累积声纳信号强度 |
| 奖励 | 当前步获得的奖励 |
| 状态 | 运行中/终止/截断 |

**摘要统计列说明：**
| 指标 | 说明 |
|------|------|
| 总步数 | 游戏执行的总步数 |
| 总奖励 | 所有步奖励的累加 |
| 平均奖励/步 | 总奖励 ÷ 总步数 |
| 最大TL(dB) | 游戏中 TL 的最大值 |
| 最小TL(dB) | 游戏中 TL 的最小值 |
| 平均TL(dB) | 游戏中 TL 的平均值 |
| 最终结果 | 游戏终止原因（胜利/被发现/超界等） |

## 常见场景和命令

```bash
# 1. 统计某次训练的游戏轮数
python step_log_analyzer.py count-games ./outputs/2026-05-13/19-08-10/step_log.txt

# 2. 回放失败的第一轮游戏
python step_log_analyzer.py replay-episode ./outputs/2026-05-13/19-08-10/step_log.txt 0

# 3. 回放成功率较高的中间轮游戏
python step_log_analyzer.py replay-episode ./outputs/2026-05-13/19-08-10/step_log.txt 80

# 4. 使用不同配置回放
python step_log_analyzer.py replay-episode ./outputs/2026-05-13/19-08-10/step_log.txt 0 \
    --config configs/c++TL_dashboard.yaml

# 5. 查看帮助信息
python step_log_analyzer.py -h
```

## 文件结构

```
GD-UUV-self/
├── step_log_analyzer.py           # 主程序文件
├── configs/
│   └── main_config.yaml           # 环境配置文件
├── env/
│   ├── env.py                     # 环境类定义
│   └── Robot.py                   # 机器人类定义
├── utils/
│   └── rich_print.py              # Rich 输出工具
├── outputs/
│   └── 2026-05-13/
│       └── 19-08-10/
│           └── step_log.txt       # 日志文件示例
└── README.md                      # 本文件
```

## 技术细节

### 日志解析算法

1. 逐行读取日志文件
2. 识别状态行（格式：5个逗号分隔的整数）
3. 提取 action 序列中的数字字符（0-6）
4. 以 `#` 作为一轮游戏的结束标记
5. 存储所有 episode 数据

### 回放算法

1. 加载配置文件创建环境实例
2. 调用 `env.reset()` 初始化环境
3. 覆盖位置属性为日志中记录的初始位置
4. 逐步执行 action 序列：
   - 调用 `env.step(action)`
   - 采集当前步的数据
   - 检查游戏是否结束
5. 生成表格和统计输出

### 性能考虑

- **count-games 模式**：O(n) 时间复杂度，n 为日志总行数
- **replay-episode 模式**：
  - 环境初始化：~5-10秒（包括加载地形、TL表、生成切片图像）
  - 单步执行：~0.01秒
  - 单轮回放（50步）：~10秒

## 故障排除

### 问题：`FileNotFoundError: 日志文件不存在`
**解决方案：**
- 检查日志文件路径是否正确
- 使用绝对路径或相对于项目根目录的路径
- 确保文件具有读权限

### 问题：`ImportError: 无法导入 env 模块`
**解决方案：**
- 确保项目根目录在 Python 路径中
- 在项目根目录运行该脚本
- 检查 `env/__init__.py` 是否存在

### 问题：`ConfigFileError: 配置文件不存在`
**解决方案：**
- 使用 `--config` 指定正确的配置文件路径
- 默认配置文件应为 `configs/main_config.yaml`
- 检查配置文件格式是否正确

### 问题：`Episode索引越界`
**解决方案：**
- 先用 count-games 模式确认总游戏轮数
- 使用 0-based 索引（第一轮为0）
- 检查日志文件是否完整

## 扩展建议

1. **导出轨迹数据**：添加 `--output-format csv|json` 选项
2. **可视化轨迹**：集成 matplotlib/plotly 进行轨迹绘图
3. **批量回放**：添加支持回放多个 episode 并生成对比报告
4. **性能分析**：添加统计不同阶段的成功率、平均奖励等
5. **缓存支持**：缓存环境初始化结果，加速连续回放

## 许可证

This tool is part of the GD-UUV project.

---

**最后更新：2026-05-13**

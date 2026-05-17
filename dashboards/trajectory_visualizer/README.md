# 强化学习轨迹回放面板 - 技术文档

本文档面向后续开发者，目标是在最小上下文下快速理解架构、运行和安全扩展本工程。

---

## 1. 工程定位

**trajectory_visualizer** 是一个实时强化学习游戏轨迹回放与可视化系统。

**核心链路：**

1. **后端 (Flask)** 扫描 `outputs/` 日志文件，提供 REST API 接口
2. **数据处理** 解析 step_log.txt 文件，执行轨迹回放，收集完整步数据
3. **前端 (Plotly)** 从 API 获取轨迹数据，渲染 3D 可视化
4. **交互** 鼠标悬停显示详细奖励信息、支持多轮游戏切换

**关键特性：**
- ✓ 自动加载最新日志
- ✓ 支持任意轮数（0-159）的游戏回放
- ✓ 3D 双轨迹展示：UUV（蓝色线）+ 敌方（红色虚线）
- ✓ 实时奖励分解显示（Stealth、Approach、TL Gradient、Area TL、Time Penalty）
- ✓ 键盘友好的 UI 设计

---

## 2. 目录结构与职责

```
trajectory_visualizer/
├── server.py              # Flask 后端主程序
├── index.html             # 页面骨架和控件容器
├── js/
│   └── main.js           # 前端可视化和交互逻辑
├── css/
│   └── style.css         # UI 样式
├── start_server.bat      # Windows 启动脚本
├── start_server.sh       # Linux/Mac 启动脚本
└── README.md            # 本文档
```

### 文件职责详解

| 文件 | 职责 |
|------|------|
| **server.py** | Flask 应用入口，提供 3 个 REST API、日志扫描、轨迹回放逻辑 |
| **index.html** | 页面 HTML 结构：日志选择、游戏选择、游戏信息、Step 详情、Plotly 容器 |
| **js/main.js** | 状态管理、API 调用、数据解析、Plotly 3D 渲染、事件绑定 |
| **css/style.css** | 深色主题、玻璃态卡片、响应式布局、奖励颜色编码 |

---

## 3. 快速开始（5 分钟）

### 3.1 启动服务

**Windows:**
```bash
cd e:\program\GD-UUV-self\dashboards\trajectory_visualizer
start_server.bat
```

**Linux/Mac:**
```bash
cd dashboards/trajectory_visualizer
bash start_server.sh
```

**手动启动:**
```bash
cd e:\program\GD-UUV-self
E:\lib\conda-env\torch_gpu\python.exe dashboards/trajectory_visualizer/server.py
```

服务启动后访问：**http://127.0.0.1:5000/**

### 3.2 基本使用

1. **自动加载** - 页面打开时自动获取最新日志和游戏总数
2. **选择游戏** - 在"加载轮数"输入框输入 0-159 之间的数字
3. **加载游戏** - 点击"▶ 加载游戏"按钮或按 Enter
4. **查看轨迹** - 3D 图表显示 UUV 和敌方的运动路径
5. **悬停查看详情** - 将鼠标移动到轨迹线上查看该步的详细奖励信息

---

## 4. 核心架构设计

### 4.1 后端 API 接口

所有 API 返回 JSON，成功状态下 `success: true`。

#### GET `/api/logs`
获取可用的日志文件列表（最新优先）

**返回示例：**
```json
{
  "success": true,
  "latest": {
    "path": "E:\\...\\outputs\\2026-05-13\\19-08-10\\step_log.txt"
  },
  "all_logs": [...]
}
```

#### POST `/api/count-games`
统计指定日志文件中的游戏轮数

**请求：**
```json
{
  "log_path": "E:\\...\\step_log.txt"
}
```

**返回：**
```json
{
  "success": true,
  "count": 160
}
```

#### POST `/api/replay-episode`
回放指定轮的游戏，返回完整轨迹数据

**请求：**
```json
{
  "log_path": "E:\\...\\step_log.txt",
  "episode_index": 10
}
```

**返回结构：**
```json
{
  "success": true,
  "trajectory": {
    "episode_info": {
      "uuv_x": 80, "uuv_y": 88, "uuv_z": 3,
      "enemy_y": 41,
      "action_count": 14
    },
    "trajectory": [
      {
        "step": 0, "action": 3,
        "uuv_x": 80, "uuv_y": 88, "uuv_z": 3,
        "enemy_x": 20, "enemy_y": 41, "enemy_z": 3,
        "reward": 0.1234, "cumulative_reward": 0.1234,
        "reward_details": {
          "sum_stealth_reward": 0.1,
          "sum_approach_reward": 0.02,
          "sum_tl_gradient_reward": -0.01,
          "sum_area_average_tl_reward": 0.01,
          "sum_fixed_time_penalty": -0.02
        },
        "acoustic_signal": 0.0,
        "result": "running"
      }
    ]
  }
}
```

### 4.2 前端状态管理（main.js）

**全局状态对象：**
```javascript
state = {
  logPath,              // 当前日志路径
  totalGames,           // 日志中的总游戏轮数
  currentEpisodeIndex,  // 当前加载的 episode 索引
  trajectoryData,       // 完整的轨迹 JSON 数据
  uuvTrajectory: [],    // 解析后的 UUV 坐标序列
  enemyTrajectory: [],  // 解析后的敌方坐标序列
  stepDetails: [],      // 每个 step 的完整数据
  selectedStep,         // 鼠标悬停的 step
  apiBaseUrl            // 后端 API 基础 URL
}
```

**主要函数模块：**

1. **日志管理** - loadLatestLog()、refreshGameCount()、loadEpisode()
2. **数据解析** - parseTrajectoryData()、updateEpisodeInfo()
3. **3D 渲染** - toUUVTrajectoryTrace()、toEnemyTrajectoryTrace()、drawTrajectory()
4. **交互处理** - initEventListeners()、updateStepDetails()

### 4.3 坐标体系（最关键）

**必须理解的约束：**

| 坐标 | 定义 | 范围 | 显示规则 |
|------|------|------|--------|
| **X** | 东西方向（公里） | 0-100 | 直接使用 |
| **Y** | 南北方向（公里） | 0-100 | 直接使用 |
| **Z** | 深度（米） | 输入为正 | **显示时取负** (z_display = -z_input) |

**敌方位置规则：**
- 敌方 X = 20 km（固定，来自 configs/main_config.yaml 的 env.enemy_x）
- 敌方 Y = 动态（随 action 改变）
- 敌方 Z = 3 m（固定深度）

**关键代码：**
```javascript
// UUV 轨迹：直接从 step.uuv_x/y/z 读取，z 显示时取负
{ x: step.uuv_x, y: step.uuv_y, z: -step.uuv_z }

// 敌方轨迹：敌方 x 固定为 20
{ x: 20, y: step.enemy_y, z: -step.enemy_z }

// Plotly 布局：z 轴范围固定为负数
zaxis.range = [-200, 0]  // 海平面(0) 到最大深度(负数)
```

---

## 5. 修改指南（常见扩展）

### 5.1 修改 UI 界面

**文件：** `index.html` + `css/style.css`

**示例：** 添加新的控制面板

```html
<!-- 在 index.html 侧栏中添加新分区 -->
<div class="panel-section">
  <h2>📊 新功能</h2>
  <div id="newFeature"></div>
</div>
```

```css
/* 在 style.css 中添加样式 */
.panel-section { /* 已有样式，复用 */ }
#newFeature { /* 新样式 */ }
```

### 5.2 修改 3D 轨迹显示

**文件：** `js/main.js` 中的 `toUUVTrajectoryTrace()` 等函数

**修改点：**

1. **颜色** - 修改 `line.color` 和 `marker.color`
   ```javascript
   line: { color: '#4cc9f0' }  // 改为其他颜色，如 '#ff6b6b'
   ```

2. **线宽** - 修改 `line.width`
   ```javascript
   line: { width: 3 }  // 改为 5 或 2
   ```

3. **标记大小** - 修改 `marker.size`
   ```javascript
   marker: { size: 4 }  // 改为 6 或 8
   ```

### 5.3 添加新的 API 端点

**文件：** `server.py`

**模板：**
```python
@app.route('/api/my-endpoint', methods=['POST'])
def my_endpoint():
    """
    功能说明
    """
    try:
        # 获取请求数据
        data = request.get_json()
        
        # 处理逻辑
        result = do_something(data)
        
        # 返回结果
        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        print_error(f"错误信息: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
```

### 5.4 修改奖励显示

**文件：** `js/main.js` 中的 `buildStepDetailsHTML()`

**修改点：** 添加新的奖励分量显示
```javascript
${rewardDetails['新的奖励字段'] !== undefined ? `
  <div class="reward-item">
    <span class="reward-name">新奖励名</span>
    <span class="reward-value ${rewardDetails['新的奖励字段'] >= 0 ? 'reward-positive' : 'reward-negative'}">
      ${formatNumber(rewardDetails['新的奖励字段'], 4)}
    </span>
  </div>
` : ''}
```

---

## 6. 数据流与执行流程

### 6.1 页面加载流程

```
┌─ DOMContentLoaded
├─ init()
│  ├─ initEventListeners()
│  ├─ loadLatestLog()
│  │  └─ GET /api/logs → state.logPath = 最新日志路径
│  └─ refreshGameCount()
│     └─ POST /api/count-games → state.totalGames = 160
│
└─ 页面就绪，用户可选择游戏
```

### 6.2 游戏加载流程

```
┌─ 用户输入轮数并点击"加载游戏"
├─ loadEpisode(episodeIndex)
│  ├─ POST /api/replay-episode → trajectoryData = 回放结果
│  ├─ parseTrajectoryData()
│  │  ├─ uuvTrajectory[] = 提取 UUV 坐标
│  │  ├─ enemyTrajectory[] = 提取敌方坐标
│  │  └─ stepDetails[] = 保存完整步数据
│  │
│  ├─ updateEpisodeInfo() → 填充游戏信息面板
│  └─ drawTrajectory()
│     ├─ toUUVTrajectoryTrace() → 蓝色轨迹线
│     ├─ toEnemyTrajectoryTrace() → 红色虚线
│     ├─ toKeyPointsTrace() → 绿色起点+红色终点
│     └─ Plotly.newPlot() → 渲染图表
│
└─ 图表显示，用户可悬停查看详情
```

### 6.3 鼠标悬停流程

```
┌─ 鼠标进入 Plotly 图表
├─ plotly_hover 事件触发
│  ├─ 检测 pointNumber（点的索引）
│  ├─ 从 stepDetails[] 查找对应 step
│  └─ state.selectedStep = 该 step 数据
│
├─ updateStepDetails()
│  └─ buildStepDetailsHTML() → 生成详情 HTML
└─ 显示在侧栏 Step 详情面板
```

---

## 7. 部署与性能优化

### 7.1 生产环境配置

**修改 server.py：**
```python
# 生产环境使用 Gunicorn（需要安装）
# pip install gunicorn

# 启动命令
gunicorn -w 4 -b 127.0.0.1:5000 server:app
```

### 7.2 大轨迹优化（>500 步）

**问题：** 绘制点过多时性能下降

**解决方案 1 - 采样：**
```javascript
// 在 drawTrajectory() 中添加采样逻辑
const sampleRate = state.uuvTrajectory.length > 500 ? 5 : 1;
const sampledUUV = state.uuvTrajectory.filter((p, i) => i % sampleRate === 0);
```

**解决方案 2 - 异步渲染：**
```javascript
// 使用 setTimeout 分离重型操作
setTimeout(() => drawTrajectory(), 100);
```

---

## 8. 常见错误与调试

### 问题：404 找不到日志文件

**原因：** `outputs/` 目录结构不符合预期  
**检查：**
```bash
# 应该是这样的结构
outputs/
├── 2026-05-13/
│   └── 19-08-10/
│       └── step_log.txt  ✓
```

**修复：** 确保日志文件存在并按日期/时间分组

---

### 问题：轨迹显示错误（Z 轴方向反了）

**原因：** 坐标体系混淆  
**检查：** main.js 中的 `toUUVTrajectoryTrace()`
```javascript
z: state.uuvTrajectory.map(p => -p.z)  // 必须取负！
```

---

### 问题：悬停无反应

**原因：** Plotly hover 事件未绑定或 stepDetails[] 为空  
**调试：**
```javascript
// 在浏览器控制台检查
console.log('stepDetails:', state.stepDetails.length);
console.log('selectedStep:', state.selectedStep);
```

---

## 9. 验证清单（修改后必做）

完成任何修改后，必须验证以下四项：

- [ ] **后端启动成功** - 访问 http://127.0.0.1:5000/api/health 返回 200 和 `{"status":"ok"}`
- [ ] **自动加载** - 刷新页面，自动显示最新日志路径和 160 个游戏
- [ ] **游戏加载** - 加载游戏 #10，应显示 14 步轨迹和蓝红双线
- [ ] **悬停交互** - 鼠标悬停到轨迹，侧栏应显示奖励分量且颜色正确（正绿、负红）

---

## 10. 技术栈

| 层 | 技术 | 版本 |
|----|------|------|
| **后端** | Flask 2.x | 2.3+ |
| **依赖** | Flask-CORS | 3.0+ |
| **前端** | Plotly.js | 2.35+ |
| **样式** | CSS 3 | 原生，无框架 |
| **运行时** | Python | 3.10.13 |

---

## 11. 文件大小与性能指标

| 文件 | 大小 | 用途 |
|------|------|------|
| server.py | ~580 lines | 后端逻辑 |
| main.js | ~660 lines | 前端逻辑 |
| index.html | ~150 lines | 页面结构 |
| style.css | ~380 lines | 样式 |
| **总计** | **~1,770 lines** | **完整应用** |

**性能基准：**
- 页面加载：< 2s
- 游戏加载（14 步）：< 1s
- 3D 渲染：< 500ms
- 悬停响应：< 50ms

---

**文档版本：1.0 | 最后更新：2026-05-13**
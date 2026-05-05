# final_tl_dashboard Agent技术文档

本文档面向后续Agent编码机器人，目标是最小上下文下快速理解并可安全扩展本工程。

## 1. 工程定位

final_tl_dashboard 是一个基于 Plotly 的三维传播损失可视化面板，采用离线预处理 + 前端纯读取渲染模式。

核心链路如下。

1. Python预处理脚本读取 TXT/CSV 与 YAML。
2. 输出前端直接消费的 JSON 数据。
3. 前端按敌方索引选择切片，渲染三维地形 surface 与 TL 点云 scatter3d。
4. 相机支持键盘平移，TL着色支持限幅。

## 2. 目录与职责

1. prepare_data.py
负责数据预处理、采样、掩码过滤、统计信息与运行时配置导出。

2. data/payload.json
主数据体。包含地形网格、我方采样点、按敌方分组的 TL 与 valid_mask。

3. data/config_runtime.json
前端渲染参数。包含颜色范围、颜色限幅、相机参数、初始敌方映射结果。

4. data/meta.json
预处理元信息与初始切片统计。

5. js/main.js
前端渲染主逻辑。完成坐标转换、trace 构建、事件绑定、重绘。

6. index.html
页面骨架与控件容器。

7. css/style.css
面板样式。

## 3. 数据输入输出契约

### 3.1 输入

1. configs/final_tl_dashboard.yaml
包含采样步长、颜色范围、颜色限幅、相机参数、初始敌方位置、数据路径。

2. configs/main_config.yaml
读取 env.enemy_x，作为敌方真实 x 坐标来源。

3. final_tl_dashboard.yaml 中配置的 terrain_txt_path
地形输入。支持三列散点格式（x,y,depth）或规则深度矩阵格式。

4. final_tl_dashboard.yaml 中配置的 tl_csv_path
TL输入。每行格式为 enemy_y,uuv_x,uuv_y,uuz_z,tl（逗号或空白分隔）。

### 3.2 输出

1. payload.json
关键字段。

- terrain.x_km, terrain.y_km, terrain.depth_m
- points.x_km, points.y_km, points.z_m
- enemy_positions
- tl_by_enemy[].tl_values 与 tl_by_enemy[].valid_mask
- stats.tl_min_db, stats.tl_max_db, stats.terrain_depth_max_m, stats.enemy_x_km

2. config_runtime.json
关键字段。

- tl_color_min_db
- tl_color_max_db
- tl_color_cap_db
- camera_position
- plotly_config
- initial_enemy

3. meta.json
关键字段。

- axes_lengths
- sampled_points
- enemy_positions
- initial_slice_stats

## 4. 坐标体系与显示规则

这是本工程最关键的约束，修改渲染逻辑时必须保持。

1. x 与 y 坐标使用真实公里值。

2. 深度/高度规则。

- 输入深度是正值 depth_m。
- 三维图 z 轴以海平面为 0，海平面以下显示负值。
- 因此前端统一采用 z = -depth 或 z = -our_z_m。

3. 敌方 x 使用 main_config.yaml 中 env.enemy_x（米）转换为公里，不再使用离散轴推导。

4. z 轴范围固定为负深度区间。

- layout.scene.zaxis.range = [-terrain_depth_max_m, 0]

## 5. 预处理主流程

入口为 prepare_data.py:main，核心阶段如下。

1. 读取并解析配置。
2. 加载地形 NPZ 与 TL NPZ。
3. 校验 tl_mean_grid 与 pair_mask 的五维形状一致性。
4. 按 sample_step_x/y/z 生成我方采样索引。
5. 在敌方二维轴 enemy_y, enemy_z 上展开循环，构建 enemy_positions。
6. 对每个敌方切片输出。

- tl_values: 当前敌方下所有我方采样点 TL 值
- valid_mask: 当前敌方下所有我方采样点有效性掩码

7. 统计全局 TL min/max 与初始切片统计。
8. 生成 payload、runtime_config、meta 并写入 data 目录。

## 6. 三维地形渲染技术流程

实现函数在 js/main.js 的 toTerrainTrace。

### 6.1 输入

1. payload.terrain.x_km
2. payload.terrain.y_km
3. payload.terrain.depth_m
4. runtimeConfig.plotly_config.terrain_opacity

### 6.2 处理

1. x, y 直接使用地形轴。
2. z 由 depth_m 逐点取负值，转换到海平面以下。
3. surfacecolor 保留 depth_m 原值用于着色。
4. 使用 Plotly surface trace，showscale 关闭。

### 6.3 输出 trace 结构

1. type = surface
2. name = 三维地形
3. visible 绑定 showTerrain
4. z = -depth
5. hovertemplate 输出真实 x/y/z

## 7. TL 点云渲染技术流程

实现链路为 buildEnemySlice -> toTlTrace。

### 7.1 敌方切片构建

1. 根据 enemyIndex 读取 tl_by_enemy[enemyIndex]。
2. 遍历 tl_values 与 valid_mask。
3. 仅保留有效且有限值。
4. 点位坐标。

- x = points.x_km[i]
- y = points.y_km[i]
- z = -points.z_m[i]

5. 颜色值执行限幅。

- color = min(tlValueDb, tlCapDb)
- tlCapDb 优先取 UI 输入 state.tlColorCapDb，其次 runtimeConfig.tl_color_cap_db，最后默认 120

6. 输出 slice。

- x/y/z/color 数组
- validCount, min, max, mean

### 7.2 点云 trace 构建

1. type = scatter3d
2. mode = markers
3. marker.color = slice.color
4. cmin 取 tl_color_min_db 或全局最小
5. cmax 使用 min(globalMax, tlCapDb) 保持色标与限幅一致
6. colorscale 来自 runtimeConfig.plotly_config.colorscale
7. hovertemplate 显示点坐标与 TL

## 8. 敌方标记与统计

1. 敌方标记函数 toEnemyMarker。

- 使用敌方真实 x/y/z
- z 采用负值显示
- marker 使用 diamond 与固定颜色

2. runMeta 与 enemyMeta。

- runMeta 展示全局统计与 TL 限幅
- enemyMeta 展示当前敌方切片统计

## 9. 相机与交互机制

1. 数据加载。

- loadAll 并发读取 payload.json 与 config_runtime.json
- 渲染初始敌方索引 runtimeConfig.initial_enemy.mapped_enemy_index

2. 重绘机制。

- 任何控件变化均调用 drawEnemyByIndex
- drawEnemyByIndex 使用 Plotly.react 重建 trace 与 scene

3. 键盘平移。

- 仅当 plot 容器拥有焦点时响应 WASD
- 修改 state.cameraCenter 的 x/y
- 取值限制在 [-1, 1]
- 每次平移后立即重绘

## 10. Agent二次开发操作规范

以下规则用于避免破坏坐标正确性和渲染一致性。

1. 不要在前端引入新的坐标缩放层，除非全链路同步修改并更新文档。

2. 任何新增字段优先加在 prepare_data.py 的 payload/runtime_config 输出中，再在 main.js 消费。

3. 修改 TL 颜色逻辑时，必须同时检查。

- buildEnemySlice 的 color 生成
- toTlTrace 的 cmin/cmax
- 侧栏统计显示是否同步

4. 如果新增敌方维度或采样策略，必须同步更新。

- payload.metadata
- meta.axes_lengths
- initial_enemy 映射逻辑

5. 预处理脚本运行命令固定使用。

```powershell
E:/lib/conda-env/torch_gpu/python.exe dashboards/final_tl_dashboard/prepare_data.py
```

6. 页面本地调试入口。

- dashboards/final_tl_dashboard/index.html

## 11. 最小验证清单

完成修改后至少验证以下四项。

1. 预处理脚本可成功执行并更新 data 下 JSON。
2. 地形为连续 surface，不是散点。
3. TL 点云颜色在阈值以上被限幅，色标上限同步变化。
4. z 轴显示范围为负深度到 0，敌方标记与点云位置一致。

# c++tl_dashboard 技术文档

该面板用于加载 output/c++tl_output 下的文本文件，并将传播损失曲面叠加到三维地形上。

## 1. 数据格式

每个 TXT 文件遵循以下格式。

1. 第 1 行：
source_x_m source_y_m source_z_m enemy_x_m enemy_y_m enemy_z_m

2. 第 2 行及以后：
receiver_depth_m enemy_r_m enemy_theta_deg tl_db

坐标与单位约定如下。

1. x、y、z、depth、r 使用米（m）。
2. theta 使用角度（deg）。
3. TL 使用 dB。
4. 前端渲染时，x/y 转为 km，z 使用负深度显示（z = -depth）。

## 2. 目录结构

c++tl_dashboard/
- prepare_data.py
- index.html
- css/style.css
- js/main.js
- data/payload.json
- data/config_runtime.json
- data/meta.json

## 3. 运行步骤

1. 预处理生成 JSON：

```powershell
E:/lib/conda-env/torch_gpu/python.exe dashboards/c++tl_dashboard/prepare_data.py
```

2. 打开页面：
- 使用 Live Server 打开 dashboards/c++tl_dashboard/index.html

## 4. 交互说明

1. 左上角下拉框选择输入文件。
2. 左侧列表点击文件可快速切换。
3. 可开关地形与 TL 点云显示。
4. 可调 TL 着色限幅（dB）。
5. 点击三维视口后支持 WASD 平移视角。

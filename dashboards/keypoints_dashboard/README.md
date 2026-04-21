# keypoints_dashboard 说明

本面板用于查看 keypoints 的三维空间分布，架构与 TL_npz_dashboard 保持一致：
1. Python 预处理脚本产出前端 JSON。
2. 前端页面只读取 JSON 进行 Plotly 渲染。

## 1. 数据来源

1. keypoints: output/keypoints/key_points.npz
2. terrain: output/bty/terrain.npz
3. 配置文件: configs/keypoint_dashboard.yaml

## 2. 运行流程

1. 先生成 keypoints（若已生成可跳过）：

```powershell
E:/lib/conda-env/torch_gpu/python.exe scripts/generate_key_points.py
```

2. 生成 dashboard 数据：

```powershell
E:/lib/conda-env/torch_gpu/python.exe dashboards/keypoints_dashboard/prepare_data.py
```

3. 打开页面：

- 使用 LiveServer 打开 dashboards/keypoints_dashboard/index.html

## 3. 输出文件

prepare_data.py 会输出到 dashboards/keypoints_dashboard/data：

1. payload.json: 地形、点云、敌方列表和统计。
2. config_runtime.json: 前端渲染参数。
3. meta.json: 预处理元信息。

## 4. 显示约定

1. x/y 单位为 km。
2. z 轴采用负深度显示（海平面以下为负）。
3. 点云颜色默认映射到 UUV 深度 z（单位 m）。
4. 通过“敌方位置”下拉框可查看不同 enemy_y 对应位置。

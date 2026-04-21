"""ACNet (Acoustic Concealment Network) 决策网络。

该文件提供用于 UUV 隐蔽突防任务的 Actor-Critic 网络实现，
网络同时接收 3D 空间输入与 6 维状态向量输入。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class ACNet(nn.Module):
    """ACNet 决策网络。

    网络结构由四部分组成：
    1. 空间分支（3D 卷积特征提取）
    2. 向量分支（MLP 特征提取）
    3. 融合层（特征拼接 + 线性映射 + LayerNorm）
    4. Actor-Critic 双输出头
    
    注意：所有推理必须在 GPU 上执行。网络不支持 CPU 推理。

    调用示例:
        >>> model = ACNet(device="cuda").to("cuda")
        >>> spatial_input = torch.randn(4, 2, 16, 16, 16, device="cuda")
        >>> state_vector = torch.randn(4, 6, device="cuda")
        >>> actor_logits, state_value = model(spatial_input, state_vector)
        >>> actor_logits.shape, state_value.shape
        (torch.Size([4, 6]), torch.Size([4, 1]))
    """

    def __init__(self, device: str = "cuda") -> None:
        """初始化 ACNet 网络结构。

        功能说明:
            构建空间分支、向量分支、融合层及 Actor-Critic 输出头。
            所有参数与缓存均被绑定至指定设备（默认 CUDA）。

        输入参数:
            device (str): 推理设备，默认为 "cuda"。
                必须为有效的 PyTorch 设备字符串（如 "cuda:0"）。
                推理阶段不支持 CPU。

        输出参数:
            无。

        调用示例:
            >>> model = ACNet(device="cuda")
            >>> model.to("cuda")
        """
        super().__init__()
        
        if not device.startswith("cuda"):
            raise ValueError(f"ACNet 仅支持 GPU 推理，指定设备: {device}。请使用 'cuda' 或 'cuda:0' 等。")
        
        self._device = device

        # Spatial Branch 输入: (B, 2, D, H, W)
        # Conv1 输出: (B, 32, D, H, W)
        self._spatial_conv1 = nn.Conv3d(in_channels=2, out_channels=32, kernel_size=5, padding=2)
        self._spatial_bn1 = nn.BatchNorm3d(num_features=32)

        # Conv2 输出: (B, 64, D, H, W)
        self._spatial_conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self._spatial_bn2 = nn.BatchNorm3d(num_features=64)

        # Conv3 输出: (B, 128, D, H, W)
        self._spatial_conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self._spatial_bn3 = nn.BatchNorm3d(num_features=128)

        # GAP 输出: (B, 128, 1, 1, 1) -> 展平后 (B, 128)
        self._spatial_gap = nn.AdaptiveAvgPool3d(output_size=1)

        # Vector Branch 输入: (B, 6)
        # MLP 输出: (B, 64)
        self._vector_mlp = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        # Fusion: 拼接后 (B, 128 + 64 = 192) -> (B, 256)
        self._fusion_fc = nn.Linear(192, 256)
        self._fusion_ln = nn.LayerNorm(256)
        self._fusion_relu = nn.ReLU(inplace=True)

        # Actor Head: (B, 256) -> (B, 6)
        self._actor_head = nn.Linear(256, 6)

        # Critic Head: (B, 256) -> (B, 1)
        self._critic_head = nn.Linear(256, 1)

    def forward(self, spatial_input: torch.Tensor, state_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行前向传播并返回策略 logits 与状态价值。

        功能说明:
            1. 对 3D 体素输入执行卷积特征提取并做全局平均池化。
            2. 对 6 维状态向量执行 MLP 编码。
            3. 将两路特征拼接后通过融合层。
            4. 由 Actor/Critic 头分别输出动作 logits 与状态价值。
            
            注意：所有输入必须已在 GPU 上。若输入在 CPU，将抛出异常。

        输入参数:
            spatial_input (torch.Tensor):
                形状为 (B, 2, D, H, W) 的 3D 输入，必须在 GPU 上。
                通道 0 表示传播损失 TL 图，通道 1 表示地形可通性数组。
            state_vector (torch.Tensor):
                形状为 (B, 6) 的状态向量，必须在 GPU 上。
                包含 (x, y, z, x_e, y_e, z_e)。

        输出参数:
            Tuple[torch.Tensor, torch.Tensor]:
                actor_logits: 形状 (B, 6)，对应 6 个动作的 logits，在 GPU 上。
                state_value: 形状 (B, 1)，对应状态价值 V(s)，在 GPU 上。

        调用示例:
            >>> model = ACNet(device="cuda").to("cuda")
            >>> spatial_input = torch.randn(2, 2, 20, 24, 28, device="cuda")
            >>> state_vector = torch.randn(2, 6, device="cuda")
            >>> actor_logits, state_value = model(spatial_input, state_vector)
        """
        # 设备检查：确保输入在 GPU 上
        if not spatial_input.is_cuda:
            raise RuntimeError(
                f"spatial_input 必须在 GPU 上执行。当前设备: {spatial_input.device}。"
                "请使用 spatial_input.to('cuda') 或在创建张量时指定 device='cuda'。"
            )
        if not state_vector.is_cuda:
            raise RuntimeError(
                f"state_vector 必须在 GPU 上执行。当前设备: {state_vector.device}。"
                "请使用 state_vector.to('cuda') 或在创建张量时指定 device='cuda'。"
            )
        
        if spatial_input.dim() != 5:
            raise ValueError(f"spatial_input 必须为 5 维张量 (B, 2, D, H, W)，当前维度: {spatial_input.dim()}")
        if spatial_input.size(1) != 2:
            raise ValueError(f"spatial_input 通道数必须为 2，当前通道数: {spatial_input.size(1)}")
        if state_vector.dim() != 2 or state_vector.size(1) != 6:
            raise ValueError(
                f"state_vector 必须为形状 (B, 6) 的 2 维张量，当前形状: {tuple(state_vector.shape)}"
            )
        if spatial_input.size(0) != state_vector.size(0):
            raise ValueError(
                "spatial_input 与 state_vector 的 batch 大小必须一致，"
                f"当前分别为 {spatial_input.size(0)} 和 {state_vector.size(0)}"
            )

        # --- 空间分支 ---
        # (B, 2, D, H, W) -> (B, 32, D, H, W)
        spatial_feature = self._spatial_conv1(spatial_input)
        spatial_feature = self._spatial_bn1(spatial_feature)
        spatial_feature = torch.relu(spatial_feature)

        # (B, 32, D, H, W) -> (B, 64, D, H, W)
        spatial_feature = self._spatial_conv2(spatial_feature)
        spatial_feature = self._spatial_bn2(spatial_feature)
        spatial_feature = torch.relu(spatial_feature)

        # (B, 64, D, H, W) -> (B, 128, D, H, W)
        spatial_feature = self._spatial_conv3(spatial_feature)
        spatial_feature = self._spatial_bn3(spatial_feature)
        spatial_feature = torch.relu(spatial_feature)

        # (B, 128, D, H, W) -> (B, 128, 1, 1, 1) -> (B, 128)
        spatial_feature = self._spatial_gap(spatial_feature)
        spatial_feature = spatial_feature.flatten(start_dim=1)

        # --- 向量分支 ---
        # (B, 6) -> (B, 64)
        vector_feature = self._vector_mlp(state_vector)

        # --- 融合层 ---
        # concat: (B, 128) + (B, 64) -> (B, 192)
        fused_feature = torch.cat([spatial_feature, vector_feature], dim=1)

        # (B, 192) -> (B, 256)
        fused_feature = self._fusion_fc(fused_feature)
        fused_feature = self._fusion_ln(fused_feature)
        fused_feature = self._fusion_relu(fused_feature)

        # --- Actor-Critic 输出 ---
        # Actor: (B, 256) -> (B, 6)
        actor_logits = self._actor_head(fused_feature)

        # Critic: (B, 256) -> (B, 1)
        state_value = self._critic_head(fused_feature)

        return actor_logits, state_value


def _run_smoke_test() -> None:
    """执行 ACNet 最小可运行测试。

    功能说明:
        构造随机输入，验证 ACNet 的 forward 能够正常执行，
        并检查 Actor 与 Critic 输出维度是否符合预期。
        强制使用 GPU；若 GPU 不可用，测试将抛出异常。

    输入参数:
        无。

    输出参数:
        无。测试结果通过终端打印展示。

    调用示例:
        >>> _run_smoke_test()
    """
    try:
        from utils.rich_print import print_error, print_info
    except Exception:  # pragma: no cover
        # 仅用于兜底，避免在独立环境运行时因导入失败中断测试。
        def print_info(message: str) -> None:
            print(f"[INFO] {message}")

        def print_error(message: str) -> None:
            print(f"[ERROR] {message}")

    # 强制使用 GPU，不做回退
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA 不可用。ACNet 仅支持 GPU 推理，请确保环境中已正确安装 CUDA 与 PyTorch GPU 版本。"
        )
    
    device = "cuda"
    model = ACNet(device=device).to(device)

    batch_size = 4
    depth, height, width = 20, 24, 28

    spatial_input = torch.randn(batch_size, 2, depth, height, width, device=device)
    state_vector = torch.randn(batch_size, 6, device=device)

    try:
        actor_logits, state_value = model(spatial_input, state_vector)
        print_info(f"模型设备: {device}")
        print_info(f"空间输入形状: {tuple(spatial_input.shape)}，设备: {spatial_input.device}")
        print_info(f"向量输入形状: {tuple(state_vector.shape)}，设备: {state_vector.device}")
        print_info(f"Actor 输出形状: {tuple(actor_logits.shape)}，期望: ({batch_size}, 6)，设备: {actor_logits.device}")
        print_info(f"Critic 输出形状: {tuple(state_value.shape)}，期望: ({batch_size}, 1)，设备: {state_value.device}")

        assert actor_logits.shape == (batch_size, 6), "Actor 输出维度不正确"
        assert state_value.shape == (batch_size, 1), "Critic 输出维度不正确"
        assert actor_logits.is_cuda, "Actor 输出必须在 GPU 上"
        assert state_value.is_cuda, "Critic 输出必须在 GPU 上"
        print_info("ACNet GPU 推理测试通过。")
    except Exception as error:
        print_error(f"ACNet 测试失败: {error}")
        raise


if __name__ == "__main__":
    _run_smoke_test()

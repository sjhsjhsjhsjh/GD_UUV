"""ACNet (Acoustic Concealment Network) 决策网络。

该文件提供用于 UUV 隐蔽突防任务的 Actor-Critic 网络实现，
网络同时接收 3D 空间输入与 6 维状态向量输入。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    """3D 残差块（Residual Block）用于增强特征提取能力。
    
    功能说明：
        实现 3D 卷积残差块，通过跳连接改善梯度流畅。
        支持步长卷积用于多尺度特征提取。
    
    输入参数：
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核大小，默认 3
        stride (int): 步长，默认 1
    
    调用示例：
        >>> block = ResidualBlock3D(32, 32)
        >>> x = torch.randn(2, 32, 16, 16, 16, device='cuda')
        >>> out = block(x)  # (2, 32, 16, 16, 16)
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        
        # 1x1 卷积用于匹配通道与空间维度
        if in_channels != out_channels or stride != 1:
            self.skip_proj = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.skip_proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行残差块前向传播。
        
        输入参数：
            x (torch.Tensor): 输入张量 (B, in_channels, D, H, W)
        
        输出参数：
            torch.Tensor: 输出张量 (B, out_channels, D', H', W')
        """
        identity = x if self.skip_proj is None else self.skip_proj(x)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = out + identity
        return out


class CrossAttentionFusion(nn.Module):
    """交叉注意力融合模块（Cross-Attention Fusion）。
    
    功能说明：
        将空间特征和状态向量特征进行注意力加权融合，
        使网络能够学习两路特征的动态权重分配。
    
    输入参数：
        spatial_dim (int): 空间特征维度，默认 128
        vector_dim (int): 向量特征维度，默认 64
        fusion_dim (int): 融合输出维度，默认 256
    
    调用示例：
        >>> fusion = CrossAttentionFusion(128, 64, 256)
        >>> spatial_feat = torch.randn(4, 128, device='cuda')
        >>> vector_feat = torch.randn(4, 64, device='cuda')
        >>> fused = fusion(spatial_feat, vector_feat)  # (4, 256)
    """
    
    def __init__(self, spatial_dim: int = 128, vector_dim: int = 64, fusion_dim: int = 256):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.vector_dim = vector_dim
        
        # Query/Key/Value 投影层
        query_hidden_dim = 64
        self.query_proj = nn.Linear(spatial_dim, query_hidden_dim)
        self.key_proj = nn.Linear(vector_dim, query_hidden_dim)
        self.value_proj = nn.Linear(vector_dim, query_hidden_dim)
        
        # 融合投影层
        self.fusion_proj = nn.Linear(spatial_dim + query_hidden_dim, fusion_dim)
        self.fusion_ln = nn.LayerNorm(fusion_dim)
        self.fusion_relu = nn.ReLU(inplace=True)
    
    def forward(self, spatial_feat: torch.Tensor, vector_feat: torch.Tensor) -> torch.Tensor:
        """执行交叉注意力融合。
        
        功能说明：
            1. 将两路特征分别投影为 Query、Key、Value
            2. 计算缩放点积注意力权重
            3. 使用权重加权融合向量特征
            4. 拼接空间特征和加权向量特征，通过融合层
        
        输入参数：
            spatial_feat (torch.Tensor): 空间特征 (B, spatial_dim)
            vector_feat (torch.Tensor): 向量特征 (B, vector_dim)
        
        输出参数：
            torch.Tensor: 融合特征 (B, fusion_dim)
        """
        # 计算 Query、Key、Value
        q = self.query_proj(spatial_feat)  # (B, 64)
        k = self.key_proj(vector_feat)     # (B, 64)
        v = self.value_proj(vector_feat)   # (B, 64)
        
        # 缩放点积注意力（Scaled Dot-Product Attention）
        # 计算每个样本的点积：q_i @ k_i^T / sqrt(d)
        scores = (torch.sum(q * k, dim=1, keepdim=True)) / (self.query_proj.out_features ** 0.5)  # (B, 1)
        weights = torch.sigmoid(scores)  # (B, 1) 使用 sigmoid 而非 softmax，保持概率范围 [0, 1]
        
        # 加权融合向量特征：weights (B, 1) * v (B, 64) -> (B, 64)
        weighted_vector = weights * v  # (B, 64)
        
        # 拼接空间特征和加权向量特征
        fused = torch.cat([spatial_feat, weighted_vector], dim=1)  # (B, 192)
        
        # 通过融合层投影到目标维度
        fused = self.fusion_proj(fused)    # (B, fusion_dim)
        fused = self.fusion_ln(fused)
        fused = self.fusion_relu(fused)
        
        return fused


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
        >>> state_vector = torch.randn(4, 7, device="cuda")
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
        # Conv1 + Residual Block 输出: (B, 32, D, H, W)
        self._spatial_conv1 = nn.Conv3d(in_channels=2, out_channels=32, kernel_size=5, padding=2)
        self._spatial_bn1 = nn.BatchNorm3d(num_features=32)
        self._spatial_residual1 = ResidualBlock3D(32, 32)
        self._spatial_dropout1 = nn.Dropout3d(p=0.3)

        # Conv2 + Residual Block + Downsample 输出: (B, 64, D/2, H/2, W/2)
        self._spatial_conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self._spatial_bn2 = nn.BatchNorm3d(num_features=64)
        self._spatial_residual2 = ResidualBlock3D(64, 64)
        self._spatial_dropout2 = nn.Dropout3d(p=0.3)

        # Conv3 + Residual Block 输出: (B, 128, D/2, H/2, W/2)
        self._spatial_conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self._spatial_bn3 = nn.BatchNorm3d(num_features=128)
        self._spatial_residual3 = ResidualBlock3D(128, 128)
        self._spatial_dropout3 = nn.Dropout3d(p=0.3)

        # GAP 输出: (B, 128, 1, 1, 1) -> 展平后 (B, 128)
        self._spatial_gap = nn.AdaptiveAvgPool3d(output_size=1)

        # Vector Branch 输入: (B, 7)
        # MLP 输出: (B, 64)
        self._vector_mlp = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self._vector_dropout = nn.Dropout(p=0.2)

        # Fusion: 使用 CrossAttentionFusion 替代简单拼接
        # 输入: (B, 128) + (B, 64) -> 输出: (B, 256)
        self._fusion = CrossAttentionFusion(spatial_dim=128, vector_dim=64, fusion_dim=256)

        # Actor Head: (B, 256) -> (B, 7)
        self._actor_head = nn.Linear(256, 7)

        # Critic Head: (B, 256) -> (B, 1)
        self._critic_head = nn.Linear(256, 1)

    def forward(self, spatial_input: torch.Tensor, state_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行前向传播并返回策略 logits 与状态价值。

        功能说明:
            1. 对 3D 体素输入执行卷积特征提取并做全局平均池化。
            2. 对 7 维状态向量执行 MLP 编码。
            3. 将两路特征拼接后通过融合层。
            4. 由 Actor/Critic 头分别输出动作 logits 与状态价值。

            注意：所有输入必须已在 GPU 上。若输入在 CPU，将抛出异常。

        输入参数:
            spatial_input (torch.Tensor):
                形状为 (B, 2, D, H, W) 的 3D 输入，必须在 GPU 上。
                通道 0 表示传播损失 TL 图，通道 1 表示地形可通性数组。
            state_vector (torch.Tensor):
                形状为 (B, 7) 的状态向量，必须在 GPU 上。
                包含 (x_uuv, y_uuv, z_uuv, y_enemy, current_tl, gradient_tl, average_tl)。

        输出参数:
            Tuple[torch.Tensor, torch.Tensor]:
                actor_logits: 形状 (B, 7)，对应 7 个动作的 logits，在 GPU 上。
                state_value: 形状 (B, 1)，对应状态价值 V(s)，在 GPU 上。

        调用示例:
            >>> model = ACNet(device="cuda").to("cuda")
            >>> spatial_input = torch.randn(2, 2, 20, 24, 28, device="cuda")
            >>> state_vector = torch.randn(2, 7, device="cuda")
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
        if state_vector.dim() != 2 or state_vector.size(1) != 7:
            raise ValueError(
                f"state_vector 必须为形状 (B, 7) 的 2 维张量，当前形状: {tuple(state_vector.shape)}"
            )
        if spatial_input.size(0) != state_vector.size(0):
            raise ValueError(
                "spatial_input 与 state_vector 的 batch 大小必须一致，"
                f"当前分别为 {spatial_input.size(0)} 和 {state_vector.size(0)}"
            )

        # --- 空间分支 ---
        # (B, 2, D, H, W) -> Conv -> BN -> ReLU -> Residual -> Dropout -> (B, 32, D, H, W)
        spatial_feature = self._spatial_conv1(spatial_input)
        spatial_feature = self._spatial_bn1(spatial_feature)
        spatial_feature = torch.relu(spatial_feature)
        spatial_feature = self._spatial_residual1(spatial_feature)
        spatial_feature = self._spatial_dropout1(spatial_feature)

        # (B, 32, D, H, W) -> Conv -> BN -> ReLU -> Residual -> Dropout -> (B, 64, D, H, W)
        spatial_feature = self._spatial_conv2(spatial_feature)
        spatial_feature = self._spatial_bn2(spatial_feature)
        spatial_feature = torch.relu(spatial_feature)
        spatial_feature = self._spatial_residual2(spatial_feature)
        spatial_feature = self._spatial_dropout2(spatial_feature)

        # (B, 64, D, H, W) -> Conv -> BN -> ReLU -> Residual -> Dropout -> (B, 128, D, H, W)
        spatial_feature = self._spatial_conv3(spatial_feature)
        spatial_feature = self._spatial_bn3(spatial_feature)
        spatial_feature = torch.relu(spatial_feature)
        spatial_feature = self._spatial_residual3(spatial_feature)
        spatial_feature = self._spatial_dropout3(spatial_feature)

        # (B, 128, D, H, W) -> GAP -> (B, 128, 1, 1, 1) -> 展平 -> (B, 128)
        spatial_feature = self._spatial_gap(spatial_feature)
        spatial_feature = spatial_feature.flatten(start_dim=1)

        # --- 向量分支 ---
        # (B, 7) -> MLP -> Dropout -> (B, 64)
        vector_feature = self._vector_mlp(state_vector)
        vector_feature = self._vector_dropout(vector_feature)

        # --- 融合层：使用 CrossAttentionFusion ---
        # (B, 128) + (B, 64) -> CrossAttention -> (B, 256)
        fused_feature = self._fusion(spatial_feature, vector_feature)

        # --- Actor-Critic 输出 ---
        # Actor: (B, 256) -> (B, 7)
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
    state_vector = torch.randn(batch_size, 7, device=device)

    try:
        actor_logits, state_value = model(spatial_input, state_vector)
        print_info(f"模型设备: {device}")
        print_info(f"空间输入形状: {tuple(spatial_input.shape)}，设备: {spatial_input.device}")
        print_info(f"向量输入形状: {tuple(state_vector.shape)}，设备: {state_vector.device}")
        print_info(f"Actor 输出形状: {tuple(actor_logits.shape)}，期望: ({batch_size}, 7)，设备: {actor_logits.device}")
        print_info(f"Critic 输出形状: {tuple(state_value.shape)}，期望: ({batch_size}, 1)，设备: {state_value.device}")

        assert actor_logits.shape == (batch_size, 7), "Actor 输出维度不正确"
        assert state_value.shape == (batch_size, 1), "Critic 输出维度不正确"
        assert actor_logits.is_cuda, "Actor 输出必须在 GPU 上"
        assert state_value.is_cuda, "Critic 输出必须在 GPU 上"
        print_info("ACNet GPU 推理测试通过。")
    except Exception as error:
        print_error(f"ACNet 测试失败: {error}")
        raise


if __name__ == "__main__":
    _run_smoke_test()

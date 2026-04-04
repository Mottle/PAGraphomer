import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_scatter import scatter_add


class PathAttention(nn.Module):
    def __init__(self, hidden_dim, temp: float = 1.0, dropout=0.2, lambda_entropy=0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.lambda_entropy = lambda_entropy

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim**-0.5
        self.temp = temp

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.orthogonal_(self.q_proj.weight)
        nn.init.orthogonal_(self.k_proj.weight)
        nn.init.orthogonal_(self.v_proj.weight)

    def attention_entropy_loss(
        self, attn_weights, extended_batch, num_graphs, eps=1e-8
    ):
        """
        计算注意力权重的熵极小化损失。
        """
        attn_weights_safe = torch.clamp(attn_weights, min=eps)

        # 计算每个路径的 -p * log(p)
        entropy_per_path = -attn_weights * torch.log(attn_weights_safe)

        # 将同一张图内所有路径的熵相加，得到每张图的全局熵
        entropy_per_graph = scatter_add(
            entropy_per_path, extended_batch, dim=0, dim_size=num_graphs
        )

        # 返回 Batch 内所有图的平均熵
        return entropy_per_graph.mean()

    def forward(self, global_feature, rw_features, batch):
        """
        参数:
            global_feature: [B, D] (图级全局特征, Query)
            rw_features: [W, N, D] 或 [N, W, D] (所有局部游走路径, Keys/Values)
            batch: [N] (节点所属图的索引)
        返回:
            fused_graph_out: [B, D] (通过全局 Attention 直接聚合出的图级特征!)
            attn_weights: [N*W] (全图路径的重要性得分)
            loss
        """
        B = global_feature.shape[0]

        # 1. 统一维度至 [N, W, D] 以对齐 batch 索引
        if rw_features.dim() == 3 and rw_features.shape[0] != batch.shape[0]:
            rw_features = rw_features.transpose(0, 1).contiguous()

        N, W, D = rw_features.shape

        # 2. 展平特征，构成全图游走池: [N*W, D]
        flat_rw = rw_features.view(-1, D)

        # 3. 核心步骤：扩展 batch 索引
        # 让这 N*W 条路径都知道自己属于哪个图 (0 ~ B-1)
        # batch 形状变化: [N] -> [N, 1] -> [N, W] -> [N*W]
        extended_batch = batch.unsqueeze(1).expand(-1, W).reshape(-1)

        # 4. 线性变换
        Q = self.q_proj(global_feature)  # [B, D]
        K = self.k_proj(flat_rw)  # [N*W, D]
        V = self.v_proj(flat_rw)  # [N*W, D]

        # 5. 计算 Attention Score
        # 用 extended_batch 把 Q 广播到 N*W 级别与 K 点积
        Q_gathered = Q[extended_batch]  # [N*W, D]
        attn_scores = (Q_gathered * K).sum(dim=-1) * self.scale * self.temp  # [N*W]
        # Clamp to prevent softmax overflow in torch_geometric.utils.softmax
        attn_scores = torch.clamp(attn_scores, min=-10.0, max=10.0)

        # 6. 图级 Softmax (在每张图的内部进行归一化)
        # attn_weights 的和在每个 graph 内部严格为 1
        attn_weights = softmax(attn_scores, extended_batch)  # [N*W]
        attn_weights_dropped = self.dropout(attn_weights)

        # 7. 聚合为图级特征
        # 将 V 按图的归属 (extended_batch) 进行加权求和，直接得到 [B, D]
        fused_graph_out = scatter_add(
            attn_weights_dropped.unsqueeze(-1) * V, extended_batch, dim=0, dim_size=B
        )

        # 8. 恢复损失计算
        loss = self.attention_entropy_loss(attn_weights, extended_batch, B)

        return fused_graph_out, attn_weights, loss * self.lambda_entropy


class LocalPathAttention(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2, lambda_entropy=0.05):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.lambda_entropy = lambda_entropy

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim**-0.5

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.orthogonal_(self.q_proj.weight)
        nn.init.orthogonal_(self.k_proj.weight)
        nn.init.orthogonal_(self.v_proj.weight)

    def attention_entropy_loss(self, attn_weights, eps=1e-8):
        """
        计算注意力权重的熵极小化损失。

        参数:
            attn_weights: Tensor, shape [N, W] 或 [B, N, W]，经过 Softmax 归一化的注意力得分。
            eps: float, 用于维持数值稳定性的极小值。

        返回:
            loss_entropy: 标量 Tensor, 表示当前 Batch 内所有节点注意力分布的平均熵。
        """
        # 限制最小值为 eps，防止 log(0) 产生 -Inf 进而导致梯度 NaN
        attn_weights_safe = torch.clamp(attn_weights, min=eps)

        # 计算每个节点的熵: -sum(p * log(p))
        # 假设输入 shape 为 [N, W]，dim=-1 即在路径维度 W 上求和
        entropy_per_node = -torch.sum(
            attn_weights * torch.log(attn_weights_safe), dim=-1
        )

        # 对所有节点取平均
        loss_entropy = entropy_per_node.mean()

        return loss_entropy

    def forward(self, global_feature, rw_features, batch):
        """
        参数:
            global_feature: Tensor, shape [B, D] (图级全局特征, B为Batch Size)
            rw_features: Tensor, shape [W, N, D] 或 [N, W, D] (节点级多路径游走特征)
            batch: Tensor, shape [N] (节点到图的映射索引)
        返回:
            final_out: Tensor, shape [N, D] (融合后的节点级特征)
            attn_weights: Tensor, shape [N, W] (用于 Motif 分析的注意力得分)
        """
        # 1. 维度广播 (Broadcasting)
        # 利用 batch 索引，将 [B, D] 的图级 Query 展开为 [N, D]
        # 确保每个节点都能获取其所属分子的宏观上下文
        Q_base = global_feature[batch]
        Q = self.q_proj(Q_base).unsqueeze(1)  # [N, 1, D]

        # 2. 游走特征维度对齐
        # 由于此前使用了 out.mean(dim=0)，说明 RUMModel 的默认输出是 [W, N, D]
        # Attention 需要以节点为 batch 进行 bmm，因此需将其转置为 [N, W, D]
        if rw_features.dim() == 3 and rw_features.shape[0] != Q_base.shape[0]:
            rw_features = rw_features.transpose(0, 1).contiguous()

        K = self.k_proj(rw_features)  # [N, W, D]
        V = self.v_proj(rw_features)  # [N, W, D]

        # 3. 计算注意力得分 (Motif Discovery 核心)
        # Q: [N, 1, D], K^T: [N, D, W] -> scores: [N, 1, W]
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale

        # 在游走路径维度上归一化，找出当前节点最重要的路径
        attn_weights = F.softmax(attn_scores, dim=-1)  # [N, 1, W]
        attn_weights_dropped = self.dropout(attn_weights)

        # 4. 局部特征聚合
        # [N, 1, W] bmm [N, W, D] -> [N, 1, D] -> [N, D]
        fused_local = torch.bmm(attn_weights_dropped, V).squeeze(1)

        # 5. 全局与局部的特征融合
        # 将节点的全局指导向量与提纯后的局部 Motif 特征相加
        # final_out = Q_base + fused_local  # [N, D]
        final_out = fused_local
        attn_weights = attn_weights.squeeze(1)
        # loss = self.attention_entropy_loss(attn_weights)
        loss = 0

        return final_out, attn_weights, loss * self.lambda_entropy

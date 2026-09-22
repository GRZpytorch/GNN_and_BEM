from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax


@dataclass
class ModelOutput:
    pred_T_unknown: torch.Tensor
    pred_q_unknown: torch.Tensor
    mu: torch.Tensor | None = None
    logvar: torch.Tensor | None = None
    latent: torch.Tensor | None = None


def _select_num_heads(out_dim: int) -> int:
    """
    与第一套网络保持一致：
    选择能够整除输出维度的最大常用多头数。
    """
    for num_heads in (8, 4, 2, 1):
        if out_dim % num_heads == 0:
            return num_heads
    return 1


class FullGraphAttentionConv(nn.Module):
    """
    从第一套网络迁移的全图多头注意力消息传递层。

    每条 source -> target 边的权重同时由：
        1) target query
        2) source key
        3) source-target 关系特征
    决定。

    另外使用 edge gate 调制每条边、每个 attention head 的消息，
    并通过 residual branch 保留节点自身状态。

    注意：
    第二套数据当前的全连接图包含 self-loop，而第一套注意力图默认
    不包含 self-loop，并依赖 residual branch 保存自身信息。
    因此这里在 forward 内部自动过滤 self-loop，使旧数据集无需重建。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        if in_dim <= 0:
            raise ValueError("in_dim must be positive")
        if out_dim <= 0:
            raise ValueError("out_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if out_dim % num_heads != 0:
            raise ValueError(
                f"out_dim={out_dim} must be divisible by "
                f"num_heads={num_heads}"
            )

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        # 与第一套网络一致的 Q / K / V 投影。
        self.query_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.key_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.value_proj = nn.Linear(in_dim, out_dim, bias=False)

        # pairwise relation:
        # [source feature, target feature, target - source]
        edge_input_dim = 3 * in_dim
        edge_hidden_dim = max(out_dim, 32)

        self.edge_bias = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim),
            nn.GELU(),
            nn.Linear(edge_hidden_dim, num_heads),
        )

        self.edge_gate = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim),
            nn.GELU(),
            nn.Linear(edge_hidden_dim, num_heads),
            nn.Sigmoid(),
        )

        self.output_proj = nn.Linear(out_dim, out_dim)

        if in_dim == out_dim:
            self.residual_proj = nn.Identity()
        else:
            self.residual_proj = nn.Linear(in_dim, out_dim)

        self.attention_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                f"x must have shape [num_nodes, in_dim], "
                f"but received {tuple(x.shape)}"
            )

        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")

        if edge_index.dtype != torch.long:
            raise TypeError("edge_index must have dtype torch.long")

        num_nodes = x.size(0)
        residual = self.residual_proj(x)

        if edge_index.size(1) == 0:
            return residual

        source = edge_index[0]
        target = edge_index[1]

        if source.min() < 0 or target.min() < 0:
            raise ValueError("edge_index contains a negative node index")

        if source.max() >= num_nodes or target.max() >= num_nodes:
            raise ValueError(
                "edge_index contains a node index outside the valid range"
            )

        # 第二套旧数据包含 self-loop；按照第一套 attention 逻辑去掉。
        non_self = source != target
        source = source[non_self]
        target = target[non_self]

        # 极端情况下若只剩一个节点，直接走 residual。
        if source.numel() == 0:
            return residual

        query = self.query_proj(x).view(
            num_nodes,
            self.num_heads,
            self.head_dim,
        )
        key = self.key_proj(x).view(
            num_nodes,
            self.num_heads,
            self.head_dim,
        )
        value = self.value_proj(x).view(
            num_nodes,
            self.num_heads,
            self.head_dim,
        )

        # target 负责 query，source 提供 key / value。
        scores = (
            query[target] * key[source]
        ).sum(dim=-1)

        scores = scores / math.sqrt(self.head_dim)

        edge_features = torch.cat(
            [
                x[source],
                x[target],
                x[target] - x[source],
            ],
            dim=-1,
        )

        scores = scores + self.edge_bias(edge_features)

        # 对每个 target 的所有入边分别做多头 softmax。
        alpha = softmax(
            scores,
            index=target,
            num_nodes=num_nodes,
        )
        alpha = self.attention_dropout(alpha)

        edge_gate = self.edge_gate(edge_features).unsqueeze(-1)

        messages = (
            alpha.unsqueeze(-1)
            * edge_gate
            * value[source]
        )

        aggregated = torch.zeros(
            (
                num_nodes,
                self.num_heads,
                self.head_dim,
            ),
            dtype=x.dtype,
            device=x.device,
        )

        aggregated.index_add_(
            dim=0,
            index=target,
            source=messages,
        )

        aggregated = aggregated.reshape(
            num_nodes,
            self.out_dim,
        )

        update = self.output_proj(aggregated)

        # 第一套 attention 的核心 residual 结构。
        return residual + update


class FullGraphAttentionBlock(nn.Module):
    """
    将第一套的 FullGraphAttentionConv 与第二套原有的
    LayerNorm + GELU + Dropout block 形式结合。

    edge_weight 参数保留，仅用于兼容第二套现有 encode/decode 调用接口；
    attention 本身不使用固定距离权重，而是学习 source-target 关系权重。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        num_heads = _select_num_heads(out_dim)

        self.conv = FullGraphAttentionConv(
            in_dim=in_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        edge_index,
        edge_weight=None,
    ):
        # edge_weight 故意不参与计算：
        # 第一套注意力使用 learned attention + relational features。
        _ = edge_weight

        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x


class DualHeadGCNVAE(nn.Module):
    """
    第二套网络结构保持不变，只把 GCN 消息传递替换为第一套 FullGraph Attention。

    保留：
        1) 3 层共享 Encoder
        2) VAE latent: mu / logvar / reparameterization
        3) T 分支 3 层 Decoder
        4) q 分支 3 层 Decoder
        5) 两个独立输出头
        6) 原有 ModelOutput / loss / train / test 接口

    类名 DualHeadGCNVAE 为兼容现有训练和测试代码而保留。
    """

    def __init__(
        self,
        in_channels: int = 9,
        hidden_channels: int = 128,
        latent_channels: int = 32,
        decoder_channels: int = 128,
        dropout: float = 0.05,
        use_variational: bool = True,
    ):
        super().__init__()
        self.use_variational = use_variational

        # ============================================================
        # Shared Encoder: 3-layer FullGraph Attention
        # ============================================================
        self.enc_gcn1 = FullGraphAttentionBlock(
            in_channels,
            hidden_channels,
            dropout=dropout,
        )
        self.enc_gcn2 = FullGraphAttentionBlock(
            hidden_channels,
            hidden_channels,
            dropout=dropout,
        )
        self.enc_gcn3 = FullGraphAttentionBlock(
            hidden_channels,
            hidden_channels,
            dropout=dropout,
        )

        # 保留第二套原来的 VAE head 结构。
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, latent_channels),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, latent_channels),
        )

        dec_in_dim = in_channels + latent_channels

        # ============================================================
        # T decoder: independent 3-layer FullGraph Attention branch
        # ============================================================
        self.t_gcn1 = FullGraphAttentionBlock(
            dec_in_dim,
            decoder_channels,
            dropout=dropout,
        )
        self.t_gcn2 = FullGraphAttentionBlock(
            decoder_channels,
            decoder_channels,
            dropout=dropout,
        )
        self.t_gcn3 = FullGraphAttentionBlock(
            decoder_channels,
            decoder_channels,
            dropout=dropout,
        )
        self.t_out = nn.Sequential(
            nn.Linear(decoder_channels, decoder_channels // 2),
            nn.GELU(),
            nn.Linear(decoder_channels // 2, 1),
        )

        # ============================================================
        # q decoder: independent 3-layer FullGraph Attention branch
        # ============================================================
        self.q_gcn1 = FullGraphAttentionBlock(
            dec_in_dim,
            decoder_channels,
            dropout=dropout,
        )
        self.q_gcn2 = FullGraphAttentionBlock(
            decoder_channels,
            decoder_channels,
            dropout=dropout,
        )
        self.q_gcn3 = FullGraphAttentionBlock(
            decoder_channels,
            decoder_channels,
            dropout=dropout,
        )
        self.q_out = nn.Sequential(
            nn.Linear(decoder_channels, decoder_channels // 2),
            nn.GELU(),
            nn.Linear(decoder_channels // 2, 1),
        )

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        if (not self.training) or (not self.use_variational):
            return mu

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(
        self,
        x,
        edge_index,
        edge_weight=None,
    ):
        h = self.enc_gcn1(x, edge_index, edge_weight)
        h = self.enc_gcn2(h, edge_index, edge_weight)
        h = self.enc_gcn3(h, edge_index, edge_weight)

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def decode_t(
        self,
        xz,
        edge_index,
        edge_weight=None,
    ):
        h = self.t_gcn1(xz, edge_index, edge_weight)
        h = self.t_gcn2(h, edge_index, edge_weight)
        h = self.t_gcn3(h, edge_index, edge_weight)
        return self.t_out(h).squeeze(-1)

    def decode_q(
        self,
        xz,
        edge_index,
        edge_weight=None,
    ):
        h = self.q_gcn1(xz, edge_index, edge_weight)
        h = self.q_gcn2(h, edge_index, edge_weight)
        h = self.q_gcn3(h, edge_index, edge_weight)
        return self.q_out(h).squeeze(-1)

    def forward(self, data) -> ModelOutput:
        x = data.x
        edge_index = data.edge_index

        # 保留第二套数据接口；attention block 内部不使用固定 edge_weight。
        edge_weight = getattr(data, "edge_weight", None)

        z, mu, logvar = self.encode(
            x,
            edge_index,
            edge_weight,
        )

        # 与第一套 decoder 一致：重新把原始节点输入 x 拼回 latent z。
        xz = torch.cat(
            [x, z],
            dim=-1,
        )

        # 第二套核心：两个独立 decoder 保持不变。
        pred_T_unknown = self.decode_t(
            xz,
            edge_index,
            edge_weight,
        )
        pred_q_unknown = self.decode_q(
            xz,
            edge_index,
            edge_weight,
        )

        return ModelOutput(
            pred_T_unknown=pred_T_unknown,
            pred_q_unknown=pred_q_unknown,
            mu=mu if self.use_variational else None,
            logvar=logvar if self.use_variational else None,
            latent=z,
        )


def assemble_full_fields(
    pred_T_unknown: torch.Tensor,
    pred_q_unknown: torch.Tensor,
    T_known: torch.Tensor,
    q_known: torch.Tensor,
    mask_T_known: torch.Tensor,
    mask_q_known: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    T_full = mask_T_known * T_known + (1.0 - mask_T_known) * pred_T_unknown
    q_full = mask_q_known * q_known + (1.0 - mask_q_known) * pred_q_unknown
    return T_full, q_full


def supervised_unknown_loss(
    pred_T_unknown: torch.Tensor,
    pred_q_unknown: torch.Tensor,
    T_true: torch.Tensor,
    q_true: torch.Tensor,
    mask_T_known: torch.Tensor,
    mask_q_known: torch.Tensor,
):
    unknown_T_mask = mask_q_known > 0.5
    unknown_q_mask = mask_T_known > 0.5

    device = pred_T_unknown.device
    dtype = pred_T_unknown.dtype

    loss_T = torch.tensor(0.0, device=device, dtype=dtype)
    loss_q = torch.tensor(0.0, device=device, dtype=dtype)

    if torch.any(unknown_T_mask):
        loss_T = F.mse_loss(pred_T_unknown[unknown_T_mask], T_true[unknown_T_mask])

    if torch.any(unknown_q_mask):
        loss_q = F.mse_loss(pred_q_unknown[unknown_q_mask], q_true[unknown_q_mask])

    total = 0.5 * (loss_T + loss_q)
    return total, {
        "loss_T": loss_T,
        "loss_q": loss_q,
        "unknown_T_mask": unknown_T_mask,
        "unknown_q_mask": unknown_q_mask,
    }


def kl_loss(mu: torch.Tensor | None, logvar: torch.Tensor | None, device=None) -> torch.Tensor:
    if mu is None or logvar is None:
        if device is None:
            device = "cpu"
        return torch.tensor(0.0, device=device)
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def compute_losses(
    data,
    output: ModelOutput,
    beta_kl: float = 1e-5,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    sup_loss, sup_aux = supervised_unknown_loss(
        pred_T_unknown=output.pred_T_unknown,
        pred_q_unknown=output.pred_q_unknown,
        T_true=data.T_true,
        q_true=data.q_true,
        mask_T_known=data.mask_T_known,
        mask_q_known=data.mask_q_known,
    )

    kl = kl_loss(output.mu, output.logvar, device=sup_loss.device)
    total_loss = sup_loss + beta_kl * kl

    T_full, q_full = assemble_full_fields(
        pred_T_unknown=output.pred_T_unknown,
        pred_q_unknown=output.pred_q_unknown,
        T_known=data.T_known,
        q_known=data.q_known,
        mask_T_known=data.mask_T_known,
        mask_q_known=data.mask_q_known,
    )

    unknown_T_mask = sup_aux["unknown_T_mask"]
    unknown_q_mask = sup_aux["unknown_q_mask"]

    T_unknown_mae = (
        torch.mean(torch.abs(T_full[unknown_T_mask] - data.T_true[unknown_T_mask]))
        if torch.any(unknown_T_mask)
        else torch.tensor(0.0, device=sup_loss.device)
    )
    q_unknown_mae = (
        torch.mean(torch.abs(q_full[unknown_q_mask] - data.q_true[unknown_q_mask]))
        if torch.any(unknown_q_mask)
        else torch.tensor(0.0, device=sup_loss.device)
    )

    info = {
        "loss_total": float(total_loss.detach().cpu()),
        "loss_sup": float(sup_loss.detach().cpu()),
        "loss_T": float(sup_aux["loss_T"].detach().cpu()),
        "loss_q": float(sup_aux["loss_q"].detach().cpu()),
        "loss_kl": float(kl.detach().cpu()),
        "T_unknown_mae": float(T_unknown_mae.detach().cpu()),
        "q_unknown_mae": float(q_unknown_mae.detach().cpu()),
    }

    tensors = {
        "T_full": T_full,
        "q_full": q_full,
        "unknown_T_mask": unknown_T_mask,
        "unknown_q_mask": unknown_q_mask,
        "latent": output.latent,
    }
    return total_loss, info, tensors

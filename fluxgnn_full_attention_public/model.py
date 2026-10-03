from __future__ import annotations

"""Full-graph attention variational graph neural network for boundary-flux reconstruction."""


from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax


@dataclass
class ModelOutput:
    q_pred: torch.Tensor
    mu: torch.Tensor | None = None
    logvar: torch.Tensor | None = None


def _select_num_heads(hidden_dim: int) -> int:
    """
    Select the largest commonly used head number that divides hidden_dim.

    This keeps the original FluxGNN constructor unchanged.
    """
    for num_heads in (8, 4, 2, 1):
        if hidden_dim % num_heads == 0:
            return num_heads

    return 1


class FullGraphAttentionConv(nn.Module):
    """
    Full-graph attention message-passing layer.

    Unlike GCNConv, this layer does not assign the same normalized weight
    to every node in a complete graph. Each source-target pair receives
    a learned attention weight determined by:

        1. the target-node query;
        2. the source-node key;
        3. the source-target relational features.

    A residual branch preserves node-specific information and prevents
    all node representations from becoming identical on a complete graph.
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

        # Node-dependent query, key, and value projections.
        self.query_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.key_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.value_proj = nn.Linear(in_dim, out_dim, bias=False)

        # Pairwise relational input:
        # [source feature, target feature, target - source].
        edge_input_dim = 3 * in_dim
        edge_hidden_dim = max(out_dim, 32)

        # Edge-dependent attention bias.
        self.edge_bias = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim),
            nn.GELU(),
            nn.Linear(edge_hidden_dim, num_heads),
        )

        # Edge-dependent message gate.
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim),
            nn.GELU(),
            nn.Linear(edge_hidden_dim, num_heads),
            nn.Sigmoid(),
        )

        self.output_proj = nn.Linear(out_dim, out_dim)

        # Preserve the original node state.
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
        """
        Parameters
        ----------
        x:
            Node feature matrix with shape [num_nodes, in_dim].

        edge_index:
            Directed edge index with shape [2, num_edges].
            edge_index[0] contains source nodes.
            edge_index[1] contains target nodes.

        Returns
        -------
        torch.Tensor
            Updated node features with shape [num_nodes, out_dim].
        """
        if x.ndim != 2:
            raise ValueError(
                f"x must have shape [num_nodes, in_dim], "
                f"but received {tuple(x.shape)}"
            )

        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError(
                "edge_index must have shape [2, num_edges]"
            )

        if edge_index.dtype != torch.long:
            raise TypeError("edge_index must have dtype torch.long")

        num_nodes = x.size(0)
        residual = self.residual_proj(x)

        # Support an empty graph without changing the model interface.
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

        # [N, H, Dh]
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

        # The target node determines the query, while the source node
        # supplies the key and value.
        #
        # scores: [E, H]
        scores = (
            query[target] * key[source]
        ).sum(dim=-1)

        scores = scores / math.sqrt(self.head_dim)

        # Pairwise relational features.
        #
        # When x contains coordinates, normals, element lengths,
        # known boundary information, masks, and source descriptors,
        # this term allows the attention weight to depend on their
        # source-target relationship.
        edge_features = torch.cat(
            [
                x[source],
                x[target],
                x[target] - x[source],
            ],
            dim=-1,
        )

        scores = scores + self.edge_bias(edge_features)

        # Normalize all incoming edges separately for each target node.
        #
        # alpha: [E, H]
        alpha = softmax(
            scores,
            index=target,
            num_nodes=num_nodes,
        )
        alpha = self.attention_dropout(alpha)

        # The gate controls the contribution of each edge to each head.
        #
        # edge_gate: [E, H, 1]
        edge_gate = self.edge_gate(edge_features).unsqueeze(-1)

        # messages: [E, H, Dh]
        messages = (
            alpha.unsqueeze(-1)
            * edge_gate
            * value[source]
        )

        # Aggregate all incoming messages at their target nodes.
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

        # The residual branch is essential for a complete graph:
        # it preserves the node's own geometric and physical state.
        return residual + update


class FluxGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()

        num_heads = _select_num_heads(hidden_dim)

        # ============================================================
        # Encoder
        # ============================================================

        self.enc1 = FullGraphAttentionConv(
            in_dim=in_dim,
            out_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.enc2 = FullGraphAttentionConv(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.enc_norm1 = nn.LayerNorm(hidden_dim)
        self.enc_norm2 = nn.LayerNorm(hidden_dim)

        # Each boundary node has a latent distribution.
        #
        # Because the encoder uses full-graph attention, the latent
        # representation at each node contains information aggregated
        # from the entire boundary while retaining its node identity.
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # ============================================================
        # Decoder
        # ============================================================

        self.dec1 = FullGraphAttentionConv(
            in_dim=in_dim + latent_dim,
            out_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.dec2 = FullGraphAttentionConv(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.dec3 = FullGraphAttentionConv(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.dec_norm1 = nn.LayerNorm(hidden_dim)
        self.dec_norm2 = nn.LayerNorm(hidden_dim)
        self.dec_norm3 = nn.LayerNorm(hidden_dim)

        # Keep the original single-output interface.
        self.out = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, data) -> ModelOutput:
        x = data.x
        edge_index = data.edge_index

        # ============================================================
        # Encoder
        # ============================================================

        h = self.enc1(x, edge_index)
        h = self.enc_norm1(h)
        h = F.gelu(h)
        h = self.dropout(h)

        h = self.enc2(h, edge_index)
        h = self.enc_norm2(h)
        h = F.gelu(h)
        h = self.dropout(h)

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        # Use stochastic sampling during training and the latent mean
        # during evaluation to obtain deterministic predictions.
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        # ============================================================
        # Decoder
        # ============================================================

        # Reintroduce the original node information so that the decoder
        # explicitly knows the position, geometry, known physical value,
        # information-type mask, and source information of each node.
        xz = torch.cat([x, z], dim=-1)

        h = self.dec1(xz, edge_index)
        h = self.dec_norm1(h)
        h = F.gelu(h)
        h = self.dropout(h)

        h = self.dec2(h, edge_index)
        h = self.dec_norm2(h)
        h = F.gelu(h)
        h = self.dropout(h)

        h = self.dec3(h, edge_index)
        h = self.dec_norm3(h)
        h = F.gelu(h)
        h = self.dropout(h)

        q_pred = self.out(h).squeeze(-1)

        return ModelOutput(
            q_pred=q_pred,
            mu=mu,
            logvar=logvar,
        )


# ================================================================
# Loss
# ================================================================

def loss_function(
    output: ModelOutput,
    target: torch.Tensor,
    beta: float = 1e-3,
):
    # Accept both [N] and [N, 1] target tensors.
    if target.ndim == 2 and target.size(-1) == 1:
        target = target.squeeze(-1)

    if output.q_pred.shape != target.shape:
        raise ValueError(
            f"Prediction shape {tuple(output.q_pred.shape)} "
            f"does not match target shape {tuple(target.shape)}"
        )

    recon_loss = F.mse_loss(
        output.q_pred,
        target,
    )

    if output.mu is not None and output.logvar is not None:
        kl_loss = -0.5 * torch.mean(
            1
            + output.logvar
            - output.mu.pow(2)
            - output.logvar.exp()
        )
    else:
        kl_loss = torch.zeros(
            (),
            dtype=target.dtype,
            device=target.device,
        )

    loss = recon_loss + beta * kl_loss

    metrics = {
        "loss": float(loss.detach().cpu()),
        "mse": float(recon_loss.detach().cpu()),
        "kl": float(kl_loss.detach().cpu()),
    }

    return loss, metrics

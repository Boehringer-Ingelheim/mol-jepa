import torch
from torch import nn
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.aggr import AttentionalAggregation


class AtomsEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 128,
        layers: int = 3,
        attn_heads: int = 4,
    ):
        super().__init__()

        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.self_attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(hidden_dim, attn_heads, batch_first=True)
                for _ in range(layers)
            ]
        )

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.pooling = AttentionalAggregation(gate_nn=gate_nn)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, batch):
        x = self.input_proj(x)

        # To avoid any loops, we convert the disjoint atoms into a padded dense tensor
        x_dense, mask = to_dense_batch(x, batch)

        for attn, norm in zip(self.self_attn_layers, self.norms):
            attn_out, _ = attn(x_dense, x_dense, x_dense, key_padding_mask=~mask)
            x_dense = norm(x_dense + attn_out)  # Residual + Norm

        x_flat = x_dense[mask]
        pooled = self.pooling(x_flat, batch)

        return self.out(pooled)

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
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(node_dim, hidden_dim)

        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                hidden_dim,
                attn_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(layers)
        ])

        # Two norms per layer: attention block + feedforward block
        self.norms1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])

        self.drop = nn.Dropout(dropout)

        # Lightweight feed-forward network
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * hidden_dim, hidden_dim),
                nn.Dropout(dropout),
            )
            for _ in range(layers)
        ])

        gate_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.pooling = AttentionalAggregation(gate_nn=gate_network)

        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, batch):
        x = self.input_proj(x)
        x_dense, mask = to_dense_batch(x, batch)

        for attn, norm1, norm2, ffn in zip(
            self.self_attn_layers, self.norms1, self.norms2, self.ffns
        ):
            attn_out, _ = attn(x_dense, x_dense, x_dense, key_padding_mask=~mask)
            x_dense = norm1(x_dense + self.drop(attn_out))
            x_dense = norm2(x_dense + ffn(x_dense))

        x_flat = x_dense[mask]
        pooled = self.pooling(x_flat, batch)

        return self.out(pooled)
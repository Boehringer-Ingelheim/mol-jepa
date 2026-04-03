from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool


class GraphEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        layers: int = 3,
        layer_type: str = "TransformerConv",
        hidden_dim: int = 128,
        output_dim: int = 128,
        activation: str = "gelu",
        dropout: float = 0.0,
        attn_heads: int = 1,
        pooling: str = "mean",
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.pooling = pooling
        self.activation = activation
        self.dropout = dropout

        in_dim = node_dim
        for _ in range(layers):
            if layer_type == "TransformerConv":
                self.convs.append(
                    TransformerConv(
                        in_dim, hidden_dim, edge_dim=edge_dim, heads=attn_heads, concat=False
                    )
                )
            self.norms.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim

        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        for conv, norm in zip(self.convs, self.norms):
            if self.activation == "gelu":
                x = F.gelu(conv(x, edge_index, edge_attr))
            elif self.activation == "relu":
                x = F.relu(conv(x, edge_index, edge_attr))
            x = norm(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        if self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "max":
            x = global_max_pool(x, batch)
        return self.proj(x)

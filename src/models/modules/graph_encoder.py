import torch
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

        ACTIVATIONS = {"gelu": F.gelu, "relu": F.relu}
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation}")
        self.act = ACTIVATIONS[activation]

        if pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = node_dim
        for _ in range(layers):
            if layer_type == "TransformerConv":
                conv = TransformerConv(
                    in_dim,
                    hidden_dim,
                    edge_dim=edge_dim,
                    heads=attn_heads,
                    concat=False,
                )
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim

        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = self.act(x)

            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

            x = norm(x)

        x = self.pool(x, batch)
        return self.proj(x)
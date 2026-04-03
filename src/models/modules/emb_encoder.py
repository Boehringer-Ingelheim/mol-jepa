from torch import nn


class EmbEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(layers):
            in_f = input_dim if i == 0 else hidden_dim
            out_f = output_dim if i == layers - 1 else hidden_dim

            self.layers.append(nn.Linear(in_f, out_f))

            if i < layers - 1:
                self.layers.append(nn.LayerNorm(out_f))
                self.layers.append(nn.GELU())
                self.layers.append(nn.Dropout(dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

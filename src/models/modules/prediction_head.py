import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class TransformerEncoderWrapper(nn.TransformerEncoderLayer):
    """TransformerEncoderLayer that can stash per-head self-attention weights.

    Set `layer.return_attn = True` before a forward pass to capture weights
    into `layer.last_attn` (shape: [B, n_heads, S, S]). Off by default so the
    fast attention path is used during training.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_attn = False
        self.last_attn = None

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        if not self.return_attn:
            return super()._sa_block(
                x, attn_mask, key_padding_mask, is_causal=is_causal
            )
        out, attn = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        self.last_attn = attn.detach()
        return self.dropout1(out)


class MultiModalPredictor(nn.Module):
    def __init__(self, hidden_dim, n_heads, n_layers, n_modalities, dropout):
        super().__init__()

        self.n_modalities = n_modalities
        self.hidden_dim = hidden_dim
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.modality_pos = nn.Embedding(n_modalities, hidden_dim)

        encoder_layer = TransformerEncoderWrapper(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )
        self.gradient_checkpointing = True

        self.modality_pred = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_modalities)]
        )

    def set_return_attn(self, flag: bool):
        for layer in self.transformer.layers:
            layer.return_attn = flag
            if not flag:
                layer.last_attn = None

    def get_attn_weights(self):
        """Return list of [B, n_heads, S, S] tensors, one per layer (or None)."""
        return [layer.last_attn for layer in self.transformer.layers]

    def forward(self, modality_embeddings, mask, missing_mask):
        B = modality_embeddings.shape[0]
        device = modality_embeddings.device

        # Positional embeddings
        indices = torch.arange(self.n_modalities, device=device)
        pos = self.modality_pos(indices).unsqueeze(0)
        z = modality_embeddings + pos  # [B, n_mods, D]

        # Mask tokens with positional embeddings added
        mask_tokens = self.mask_token + pos  # [1, n_mods, D]
        mask_tokens = mask_tokens.expand(B, -1, -1)  # [B, n_mods, D]

        # Apply both masks
        prediction_mask_exp = mask.unsqueeze(-1)
        missing_mask_exp = missing_mask.unsqueeze(-1)
        z = torch.where(prediction_mask_exp, mask_tokens, z)
        z = torch.where(missing_mask_exp, mask_tokens, z)

        # Run transformer with gradient checkpointing
        if self.gradient_checkpointing and self.training:
            for layer in self.transformer.layers:
                z = checkpoint(layer, z, None, missing_mask, use_reentrant=False)
            if self.transformer.norm is not None:
                z = self.transformer.norm(z)
            out = z
        else:
            out = self.transformer(z, src_key_padding_mask=missing_mask)

        predictions = torch.stack(
            [self.modality_pred[i](out[:, i]) for i in range(self.n_modalities)], dim=1
        )
        return predictions, out

from __future__ import annotations

import torch
import torch.nn as nn

from backbones.modules_filter import (
    TxaFilterEnsembleTorch,
    RxaFilterEnsembleTorch,
)


def _pick_n_heads(feat_size: int, preferred: int = 4) -> int:
    for n_heads in (preferred, 2, 1):
        if feat_size % n_heads == 0:
            return n_heads
    return 1


class TemporalMixTransformer(nn.Module):
    """
    Transformer over time on the joint I/Q × channel vector (size 2C).

    Prefer seq_len <= attn_chunk (default 512) for full attention.
    Longer sequences are split into non-overlapping chunks.
    """

    def __init__(
        self,
        n_channels: int,
        n_layers: int = 1,
        n_heads: int | None = None,
        d_ff: int | None = None,
        attn_chunk: int = 512,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.feat_size = 2 * n_channels
        self.attn_chunk = attn_chunk
        n_heads = n_heads or _pick_n_heads(self.feat_size)
        if self.feat_size % n_heads != 0:
            raise ValueError(
                f"feat_size ({self.feat_size}) must be divisible by n_heads ({n_heads})"
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feat_size,
            nhead=n_heads,
            dim_feedforward=d_ff or 4 * self.feat_size,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_batch, seq_len, _, _ = x.shape
        seq = x.reshape(n_batch, seq_len, self.feat_size)
        chunk = self.attn_chunk
        if seq_len <= chunk:
            out = self.transformer(seq)
        else:
            pieces = [
                self.transformer(seq[:, t0 : t0 + chunk])
                for t0 in range(0, seq_len, chunk)
            ]
            out = torch.cat(pieces, dim=1)
        return out.view(n_batch, seq_len, self.n_channels, 2)


class TemporalTransformer(nn.Module):
    def __init__(self, seq_len, tx_filt_size, rx_filt_size, n_channels):
        super().__init__()

        self.txa_filter_layers = TxaFilterEnsembleTorch(
            n_channels, tx_filt_size, seq_len
        )
        self.temporal_attn = TemporalMixTransformer(n_channels)
        self.rxa_filter_layers = RxaFilterEnsembleTorch(
            n_channels, rx_filt_size, seq_len
        )

    def forward(self, x, h_0=None):
        x = self.txa_filter_layers(x)
        x = self.temporal_attn(x)
        return self.rxa_filter_layers(x)

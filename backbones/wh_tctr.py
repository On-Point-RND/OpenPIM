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


class ChannelAttnTransformer(nn.Module):
    """
    Lightweight Transformer encoder over RF channels at each time step.

    Input / output: (B, T, C, 2). Self-attention runs along C; T is folded into batch.
    """

    def __init__(
        self,
        n_channels: int,
        d_embed: int = 16,
        n_layers: int = 1,
        n_heads: int | None = None,
        d_ff: int | None = None,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.d_embed = d_embed
        n_heads = n_heads or _pick_n_heads(d_embed)
        if d_embed % n_heads != 0:
            raise ValueError(
                f"d_embed ({d_embed}) must be divisible by n_heads ({n_heads})"
            )

        self.in_proj = nn.Linear(2, d_embed)
        self.out_proj = nn.Linear(d_embed, 2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_embed,
            nhead=n_heads,
            dim_feedforward=d_ff or 2 * d_embed,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_batch, seq_len, n_channels, _ = x.shape
        seq = x.reshape(n_batch * seq_len, n_channels, 2)
        residual = seq
        h = self.in_proj(seq)
        h = self.transformer(h)
        h = self.out_proj(h)
        out = residual + h
        return out.view(n_batch, seq_len, n_channels, 2)


class TemporalMixTransformer(nn.Module):
    """
    Transformer encoder over time on the full I/Q cross-channel vector.

    Input / output: (B, T, C, 2). At each step the model sees all C channels jointly.
    """

    def __init__(
        self,
        n_channels: int,
        n_layers: int = 1,
        n_heads: int | None = None,
        d_ff: int | None = None,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.feat_size = 2 * n_channels
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
        out = self.transformer(seq)
        return out.view(n_batch, seq_len, self.n_channels, 2)


class TemporalChannelTransformer(nn.Module):
    def __init__(self, seq_len, tx_filt_size, rx_filt_size, n_channels):
        super().__init__()

        self.txa_filter_layers = TxaFilterEnsembleTorch(
            n_channels, tx_filt_size, seq_len
        )
        self.channel_attn = ChannelAttnTransformer(n_channels)
        self.temporal_attn = TemporalMixTransformer(n_channels)
        self.rxa_filter_layers = RxaFilterEnsembleTorch(
            n_channels, rx_filt_size, seq_len
        )

    def forward(self, x, h_0=None):
        x = self.txa_filter_layers(x)
        x = self.channel_attn(x)
        x = self.temporal_attn(x)
        return self.rxa_filter_layers(x)

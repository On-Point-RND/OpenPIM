from __future__ import annotations

import torch.nn as nn

from backbones.modules_filter import (
    TxaFilterEnsembleTorch,
    RxaFilterEnsembleTorch,
)
from backbones.wh_ctr import ChannelAttnTransformer
from backbones.wh_ttr import TemporalMixTransformer


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

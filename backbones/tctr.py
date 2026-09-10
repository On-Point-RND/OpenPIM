import torch.nn as nn

from backbones.wh_ctr import ChannelAttnTransformer
from backbones.wh_ttr import TemporalMixTransformer


class MultiChannelPureTCTR(nn.Module):
    """
    Channel then temporal Transformer mixers without TX/RX FIR (non-WH).

    Same cores as wh_tctr. Input: (B, T, C, 2). Output matches targets layout
    used in training: (T, C, 2) when B == 1, otherwise (B, T, C, 2).
    """

    def __init__(self, seq_len, tx_filt_size, rx_filt_size, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.channel_attn = ChannelAttnTransformer(n_channels)
        self.temporal_attn = TemporalMixTransformer(n_channels)

    def forward(self, x, h_0=None):
        n_batch, seq_len, _, _ = x.shape
        out = self.channel_attn(x)
        out = self.temporal_attn(out)
        out = out.view(n_batch, seq_len, self.n_channels, 2)

        # prepare_batch drops the batch dim from targets; match that layout.
        if n_batch == 1:
            out = out.squeeze(0)
        if self.n_channels == 1:
            out = out.squeeze(-2)
        return out

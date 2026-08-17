import torch
import torch.nn as nn

from backbones.wh_rnn import ChannelMixRNN


class MultiChannelPureRNN(nn.Module):
    """
    GRU over the full I/Q cross-channel vector at each time step, without TX/RX FIR.

    Input: (B, T, C, 2). Output matches targets layout used in training: (T, C, 2)
    when B == 1, otherwise (B, T, C, 2).
    """

    def __init__(self, seq_len, tx_filt_size, rx_filt_size, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.channel_mix = ChannelMixRNN(n_channels)

    def forward(self, x, h_0=None):
        n_batch, seq_len, _, _ = x.shape
        out = self.channel_mix(x)
        out = out.view(n_batch, seq_len, self.n_channels, 2)

        # prepare_batch drops the batch dim from targets; match that layout.
        if n_batch == 1:
            out = out.squeeze(0)
        if self.n_channels == 1:
            out = out.squeeze(-2)
        return out

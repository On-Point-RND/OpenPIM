import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.modules_filter import (
    TxaFilterEnsembleTorch,
    RxaFilterEnsembleTorch,
)


class ChannelMixRNN(nn.Module):
    """
    Single-layer GRU over time on the full I/Q cross-channel vector at each step.

    Input / output: (B, T, C, 2). At time t the RNN sees all C channels jointly.
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.feat_size = 2 * n_channels
        self.rnn = nn.GRU(
            input_size=self.feat_size,
            hidden_size=self.feat_size,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, h_0=None) -> torch.Tensor:
        n_batch, seq_len, n_channels, _ = x.shape
        seq = x.reshape(n_batch, seq_len, self.feat_size)
        out, _ = self.rnn(seq, h_0)
        return out.view(n_batch, seq_len, self.n_channels, 2)


class MultiChannelRNN(nn.Module):
    def __init__(self, seq_len, tx_filt_size, rx_filt_size, n_channels):
        super().__init__()

        self.txa_filter_layers = TxaFilterEnsembleTorch(
            n_channels, tx_filt_size, seq_len
        )
        self.channel_mix = ChannelMixRNN(n_channels)
        self.rxa_filter_layers = RxaFilterEnsembleTorch(
            n_channels, rx_filt_size, seq_len
        )

    def forward(self, x, h_0=None):
        x = self.txa_filter_layers(x)
        x = self.channel_mix(x)
        return self.rxa_filter_layers(x)

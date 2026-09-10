import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.modules_filter import (
    TxaFilterEnsembleTorch,
    RxaFilterEnsembleTorch,
)


class ChannelMixEnsemble(nn.Module):
    """
    Channel-axis Conv1d mixer. Input / output: (B, T, C, 2).

      (B, T, C, 2)
        → (B·T, 2, C)              # I/Q = in_channels, C = conv axis
        → Conv1d(2 → 2C, kernel=C) # one full-span window; 2C independent filters
        → (B·T, 2C, 1)
        → (B, T, C, 2)

    No temporal mixing. Default PyTorch weight init (no identity).
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.mix = nn.Conv1d(
            in_channels=2,
            out_channels=2 * n_channels,
            kernel_size=n_channels,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_batch, seq_len, n_channels, _ = x.shape
        # (B, T, C, 2) -> (B·T, 2, C)
        xc = x.permute(0, 1, 3, 2).reshape(n_batch * seq_len, 2, n_channels)
        y = self.mix(xc).squeeze(-1)  # (B·T, 2C)
        return y.reshape(n_batch, seq_len, n_channels, 2)


class WhCnn1d(nn.Module):
    def __init__(self, seq_len, tx_filt_size, rx_filt_size, n_channels):
        super().__init__()

        self.txa_filter_layers = TxaFilterEnsembleTorch(
            n_channels, tx_filt_size, seq_len
        )
        self.channel_mix = ChannelMixEnsemble(n_channels)
        self.rxa_filter_layers = RxaFilterEnsembleTorch(
            n_channels, rx_filt_size, seq_len
        )

    def forward(self, x, h_0=None):
        x = self.txa_filter_layers(x)
        x = self.channel_mix(x)
        x = F.gelu(x)
        return self.rxa_filter_layers(x)

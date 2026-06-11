import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.modules_filter import (
    TxaFilterEnsembleTorch,
    RxaFilterEnsembleTorch,
)


class ChannelMixEnsemble(nn.Module):
    """
    2D conv over (time, RF channel): short temporal window and full channel mix.

    Input / output: (B, T, C, 2). Conv2d kernel (k_t, C) with padding (k_t//2, 0)
    keeps T and C unchanged in the output layout.
    """

    def __init__(self, n_channels: int, mix_time_kernel_size: int = 3):
        super().__init__()
        if mix_time_kernel_size % 2 != 1:
            raise ValueError("mix_time_kernel_size must be odd")
        self.n_channels = n_channels
        self.mix_time_kernel_size = mix_time_kernel_size
        self.mix = nn.Conv2d(
            in_channels=2,
            out_channels=2 * n_channels,
            kernel_size=(mix_time_kernel_size, n_channels),
            padding=(mix_time_kernel_size // 2, 0),
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_batch, seq_len, n_channels, _ = x.shape
        xc = x.permute(0, 3, 1, 2)
        y = self.mix(xc).squeeze(-1).permute(0, 2, 1)
        return y.view(n_batch, seq_len, n_channels, 2)


class MultiChannelConv(nn.Module):
    def __init__(
        self,
        seq_len,
        tx_filt_size,
        rx_filt_size,
        n_channels,
        mix_time_kernel_size: int = 3,
    ):
        super().__init__()

        self.txa_filter_layers = TxaFilterEnsembleTorch(
            n_channels, tx_filt_size, seq_len
        )
        self.channel_mix = ChannelMixEnsemble(
            n_channels, mix_time_kernel_size=mix_time_kernel_size
        )
        self.rxa_filter_layers = RxaFilterEnsembleTorch(
            n_channels, rx_filt_size, seq_len
        )

    def forward(self, x, h_0=None):
        x = self.txa_filter_layers(x)
        x = F.gelu(x)
        x = self.channel_mix(x)
        x = F.gelu(x)
        return self.rxa_filter_layers(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.modules_filter import (
    TxaFilterEnsembleTorch,
    RxaFilterEnsembleTorch,
)


class ChannelMixEnsemble(nn.Module):
    """
    Same channel mix as wh_cnn1d, plus a short temporal window.

    Input / output: (B, T, C, 2).

      (B, T, C, 2)
        → (B, 2, T, C)                    # I/Q = in_channels
        → Conv2d(2 → 2C, kernel=(k_t, C)) # time window k_t + full channel span
        → (B, 2C, T, 1)                   # pad only on time; channel axis collapses
        → (B, T, C, 2)

    Default k_t=7. Default PyTorch init, bias=True.
    """

    def __init__(self, n_channels: int, mix_time_kernel_size: int = 7):
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
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_batch, seq_len, n_channels, _ = x.shape
        # (B, T, C, 2) -> (B, 2, T, C)
        xc = x.permute(0, 3, 1, 2)
        y = self.mix(xc).squeeze(-1)  # (B, 2C, T)
        return y.permute(0, 2, 1).reshape(n_batch, seq_len, n_channels, 2)


class WhCnn2d(nn.Module):
    def __init__(
        self,
        seq_len,
        tx_filt_size,
        rx_filt_size,
        n_channels,
        mix_time_kernel_size: int = 7,
    ):
        super().__init__()

        self.txa_filter_layers = TxaFilterEnsembleTorch(
            n_channels, tx_filt_size, seq_len
        )
        self.channel_mix = ChannelMixEnsemble(
            n_channels,
            mix_time_kernel_size=mix_time_kernel_size,
        )
        self.rxa_filter_layers = RxaFilterEnsembleTorch(
            n_channels, rx_filt_size, seq_len
        )

    def forward(self, x, h_0=None):
        x = self.txa_filter_layers(x)
        x = self.channel_mix(x)
        x = F.gelu(x)
        return self.rxa_filter_layers(x)

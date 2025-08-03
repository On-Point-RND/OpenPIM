import torch
import torch.nn as nn

from backbones.common_modules import (
    TxaFilterComplexTorch,
    RxaFilterComplexTorch,
)


class CondNlinCore(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        # Learnable scale parameter initialized to 1 for each channel
        self.scale = nn.Parameter(torch.ones(n_channels))  # Shape: (n_channels,)

    def forward(self, x, h_0=None):
        # Calculate nonlinearity coefficient
        amp = x[..., 0].pow(2) + x[..., 1].pow(2)
        amp = (amp.squeeze(-1) * self.scale).unsqueeze(-1).unsqueeze(-1)
        nlin_distorted = amp * x
        return nlin_distorted


class LinearConductive(nn.Module):
    def __init__(self, in_seq_size, out_seq_size, n_channels):
        super().__init__()
        self.out_seq_size = out_seq_size
        self.n_channels = n_channels
        self.txa_filter_layers = TxaFilterComplexTorch(
            n_channels, in_seq_size, out_seq_size
        )

        self.nlin_layer = CondNlinCore(n_channels)

        self.rxa_filter_layers = RxaFilterComplexTorch(n_channels, out_seq_size)

        self.bn_output = nn.BatchNorm1d(n_channels)  # For complex output

    def forward(self, x, h_0=None):
        filtered_x = self.txa_filter_layers(x)
        nonlin_output = self.nlin_layer(filtered_x)
        filt_rxa = self.rxa_filter_layers(nonlin_output)
        # output = self.bn_output(filt_rxa)
        return filt_rxa.squeeze(2)

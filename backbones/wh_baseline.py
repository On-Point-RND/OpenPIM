import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.modules_filter import (
    TxaFilterEnsembleTorch,
    RxaFilterEnsembleTorch,
)


class MultiChannelFIR(nn.Module):
    def __init__(self, seq_len, tx_filt_size, rx_filt_size, n_channels):
        super().__init__()

        self.txa_filter_layers = TxaFilterEnsembleTorch(
            n_channels, tx_filt_size, seq_len
        )

        self.rxa_filter_layers = RxaFilterEnsembleTorch(
            n_channels, rx_filt_size, seq_len
        )

    def forward(self, x, h_0=None):
        x = self.txa_filter_layers(x)
        z = torch.complex(x[..., 0], x[..., 1])
        z = z * F.gelu(z.abs())
        x = torch.stack((z.real, z.imag), dim=-1)
        return self.rxa_filter_layers(x)

import torch.nn as nn

from backbones.common_modules import (
    TxaFilterEnsemble, RxaFilterEnsemble
)


class ExternalNlinCore(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.pre_nlin_mix = nn.Linear(
            2 * n_channels,
            2 * n_channels,
            bias=False
        )

    def forward(self, x, h_0=None):
        b, s, c, _ = x.shape
        mixed_x = self.pre_nlin_mix(
            x.reshape(b * s, c * 2)
        )
        mixed_x = mixed_x.reshape(b, s, c, 2)
        # Calculate nonlinearity coefficient
        amp = mixed_x[..., 0].pow(2) + mixed_x[..., 1].pow(2)
        amp = amp.unsqueeze(-1)
        # Apply nonlinearity to already mixed signals to distort them
        nlin_distorted = amp * mixed_x
        return nlin_distorted


class LinearExternal(nn.Module):
    def __init__(self, in_seq_size, out_seq_size, n_channels):
        super().__init__()
        self.in_seq_size = in_seq_size
        self.out_seq_size = out_seq_size
        self.n_channels = n_channels

        self.txa_filter_layers = TxaFilterEnsemble(
            n_channels, in_seq_size, out_seq_size
        )

        self.nlin_core = ExternalNlinCore(
            n_channels
        )

        self.rxa_filter_layers = RxaFilterEnsemble(
            n_channels, out_seq_size
        )

        self.bn_output = nn.BatchNorm1d(n_channels)

    def forward(self, x, h_0=None):
        filtered_x = self.txa_filter_layers(x)
        nonlin_output = self.nlin_core(filtered_x)
        output = self.rxa_filter_layers(nonlin_output)
        output = self.bn_output(output)
        return output

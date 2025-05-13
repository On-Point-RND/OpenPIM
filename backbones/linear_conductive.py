import torch.nn as nn

from backbones.common_modules import (
    TxaFilterEnsemble, RxaFilterEnsemble
)


class CondNlinCore(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels

    def forward(self, x, h_0=None):        
        # Calculate nonlinearity coefficient
        amp = x[..., 0].pow(2) + x[..., 1].pow(2)

        # Apply nonlinearity to distort signals
        nlin_distorted = amp.unsqueeze(-1) * x

        return nlin_distorted


class LinearConductive(nn.Module):
    def __init__(self, in_seq_size, out_seq_size, n_channels):
        super().__init__()
        self.in_seq_size = in_seq_size
        self.out_seq_size = out_seq_size
        self.n_channels = n_channels

        self.txa_filter_layers = TxaFilterEnsemble(
            n_channels, in_seq_size, out_seq_size
        )

        self.nlin_core = CondNlinCore(
            n_channels
        )

        self.rxa_filter_layers = RxaFilterEnsemble(
            n_channels, out_seq_size
        )

        self.bn_output = nn.BatchNorm1d(n_channels)  # For complex output

    def forward(self, x, h_0=None):
        filtered_x = self.txa_filter_layers(x)
        nonlin_output = self.nlin_core(filtered_x)
        output = self.rxa_filter_layers(nonlin_output)
        output = self.bn_output(output)
        return output

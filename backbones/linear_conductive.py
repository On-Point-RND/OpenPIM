import torch.nn as nn

from backbones.common_modules import (
    TxaFilterEnsembleTorch, RxaFilterEnsembleTorch, MediumSimulation
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
        self.out_seq_size = out_seq_size
        self.n_channels = n_channels
        self.simulate_medium = False
        self.blank_medium = False
        self.medium_sim_size = 1
        self.out_txa_filter_seq_size = out_seq_size
        if self.simulate_medium:
            self.out_txa_filter_seq_size += self.medium_sim_size - 1
        self.txa_filter_layers = TxaFilterEnsembleTorch(
            n_channels, in_seq_size, self.out_txa_filter_seq_size
        )

        self.nlin_layer = CondNlinCore(
            n_channels
        )

        self.medium_simulation_layer = MediumSimulation(
            n_channels, self.medium_sim_size
        )

        self.rxa_filter_layers = RxaFilterEnsembleTorch(
            n_channels, out_seq_size
        )

        self.bn_output = nn.BatchNorm1d(n_channels)  # For complex output

    def forward(self, x, h_0=None):
        filtered_x = self.txa_filter_layers(x)
        nonlin_output = self.nlin_layer(filtered_x)
        if self.simulate_medium:
            if not self.blank_medium:
                nonlin_output = self.medium_simulation_layer(nonlin_output)
            else:
                nonlin_output = nonlin_output[:, : self.out_seq_size, ...]
        filt_rxa = self.rxa_filter_layers(nonlin_output)
        output = self.bn_output(filt_rxa)
        return output

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.utils as utils

from backbones.common_modules import (
    TxaFilterEnsembleTorch,
    RxaFilterEnsembleTorch,
    MediumSimulation,
)


class AmplitudeAwareNonlin(nn.Module):
    def __init__(self, hidden_size=16, num_layers=2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = 1 if i == 0 else hidden_size
            out_dim = 1 if i == num_layers - 1 else hidden_size
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)
        # Initialize to identity: g(|x|²) ≈ |x|²
        with torch.no_grad():
            self.net[-1].weight.zero_()  # Zero final weight
            self.net[-1].bias.fill_(1.0)  # Set bias=1 → g(a)=1*a

    def forward(self, x):
        # x: [..., 2]
        amp_sq = (x[..., 0] ** 2 + x[..., 1] ** 2).unsqueeze(-1)  # [..., 1]
        scaling = self.net(amp_sq)  # [..., 1]
        return x * scaling  # Distort: (x_real, x_imag) * g(|x|²)


class LearnableNlinCore(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.nonlin = AmplitudeAwareNonlin()

    def forward(self, x):
        return self.nonlin(x)


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

        self.nlin_layer = LearnableNlinCore(n_channels)

        self.medium_simulation_layer = MediumSimulation(
            n_channels, self.medium_sim_size
        )

        self.rxa_filter_layers = RxaFilterEnsembleTorch(n_channels, out_seq_size)

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

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.utils as utils

from backbones.common_modules import (
    TxaFilterEnsembleTorch,
    RxaFilterEnsembleTorch,
    MediumSimulation,
)


class PhaseAwareNonlin(nn.Module):
    def __init__(self, hidden_size=16, num_layers=2):
        super().__init__()
        layers = []
        # Input: [I, Q, |x|] (3 features)
        for i in range(num_layers):
            in_dim = 3 if i == 0 else hidden_size
            out_dim = 2 if i == num_layers - 1 else hidden_size
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

        # Initialize to identity: f(x) ≈ x
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.fill_(1.0)

    def forward(self, x):
        amps = torch.norm(x, dim=-1, keepdim=True)
        # Features: I, Q, amplitude
        features = torch.cat([x, amps], dim=-1)
        return self.net(features)


class LearnableNlinCore(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.nonlin = PhaseAwareNonlin()

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

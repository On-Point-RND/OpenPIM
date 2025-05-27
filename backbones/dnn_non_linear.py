import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.common_modules import (
    TxaFilterEnsembleTorch,
    RxaFilterEnsembleTorch,
    MediumSimulation,
)


class MatrixNonlinLayer(nn.Module):
    def __init__(self, n_channels, nonlinearity="silu"):
        super().__init__()
        # Store number of channels explicitly
        self.n_channels = n_channels

        # Initialize matrices as identity matrices
        self.linear = nn.Linear(2 * n_channels, 2 * n_channels, bias=True)

        # Set non-linearity
        self.nlin = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "silu": nn.SiLU(),
            "none": nn.Identity(),
        }[nonlinearity]

    def forward(self, x):
        batch, time = x.shape[0], x.shape[1]
        x_flat = x.view(batch * time, -1)  # Shape: (B*T, C, 2)
        transformed = self.linear(x_flat)
        transformed = transformed.view(batch, time, self.n_channels, 2)
        return self.nlin(transformed)


class LearnableNlinCore(nn.Module):
    def __init__(self, n_channels, num_layers=10, nonlinearity="silu"):
        super().__init__()
        # Store actual number of channels
        self.n_channels = n_channels

        layers = []
        for _ in range(num_layers):
            layers.append(MatrixNonlinLayer(n_channels, nonlinearity))
        self.model = nn.Sequential(*layers)

    def forward(self, x, h_0=None):
        # Additional input validation
        return self.model(x)


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

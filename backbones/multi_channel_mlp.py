import torch.nn as nn
import torch.nn.init as init

from backbones.common_modules import (
    TxaFilterEnsembleTorch,
    RxaFilterEnsembleTorch,
)


class SingleLayerPerceptron(nn.Module):
    def __init__(self, n_channels, nonlinearity="silu"):
        super().__init__()
        self.n_channels = n_channels

        # Linear layer: input and output are both 2 * n_channels
        self.linear = nn.Linear(2 * n_channels, 2 * n_channels, bias=True)

        # Initialize weights as identity matrix
        self._initialize_as_identity()

        # Set non-linearity
        self.nlin = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "silu": nn.SiLU(),
            "none": nn.Identity(),
        }[nonlinearity]

    def _initialize_as_identity(self):
        init.eye_(self.linear.weight)
        # Optional: zero out the bias
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, x):
        batch, time = x.shape[0], x.shape[1]
        x_flat = x.view(batch * time, -1)  # Shape: (B*T, C*2)
        transformed = self.linear(x_flat)
        transformed = transformed.view(batch, time, self.n_channels, 2)
        return self.nlin(transformed)


class NlinCore(nn.Module):
    def __init__(self, n_channels, num_layers=5, nonlinearity="silu"):
        super().__init__()
        self.n_channels = n_channels
        layers = []
        for _ in range(num_layers):
            layers.append(SingleLayerPerceptron(n_channels, nonlinearity))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MultiChannelMLP(nn.Module):
    def __init__(self, in_seq_size, out_seq_size, n_channels):
        super().__init__()
        self.n_channels = n_channels

        self.txa_filter_layers = TxaFilterEnsembleTorch(
            n_channels, in_seq_size, out_seq_size
        )

        self.nlin_layer = NlinCore(n_channels)

        self.rxa_filter_layers = RxaFilterEnsembleTorch(
            n_channels, out_seq_size
        )

        self.bn_output = nn.BatchNorm1d(n_channels)

    def forward(self, x, h_0=None):
        filtered_x = self.txa_filter_layers(x)
        nonlin_output = self.nlin_layer(filtered_x)
        filt_rxa = self.rxa_filter_layers(nonlin_output)
        output = self.bn_output(filt_rxa)
        return output

import torch.nn as nn

from backbones.modules_filter import (
    TxaFilterEnsembleTorch,
    RxaFilterEnsembleTorch,
)

from backbones.modules_mlp import SingleLayerPerceptron


class NlinCore(nn.Module):
    def __init__(self, n_channels, num_layers, nonlinearity="silu"):
        super().__init__()
        self.n_channels = n_channels
        layers = []
        for _ in range(num_layers):
            layers.append(SingleLayerPerceptron(n_channels, nonlinearity))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch, time, n_channels, 2)
        batch, time = x.shape[0], x.shape[1]
        # Flatten for processing: (batch * time, C*2)
        x_flat = x.view(batch * time, -1)
        transformed = self.model(x_flat)
        # Reshape back: (batch, time, n_channels, 2)
        transformed = transformed.view(batch, time, self.n_channels, 2)
        return transformed


class MultiChannelMLP(nn.Module):
    def __init__(self, in_seq_size, out_seq_size, n_channels):
        super().__init__()
        num_mlp_layers = 3

        self.txa_filter_layers = TxaFilterEnsembleTorch(
            n_channels, in_seq_size, out_seq_size
        )

        self.nlin_layer = NlinCore(
            n_channels, num_mlp_layers
        )

        self.rxa_filter_layers = RxaFilterEnsembleTorch(
            n_channels, out_seq_size
        )

    def forward(self, x, h_0=None):
        filtered_x = self.txa_filter_layers(x)
        nonlin_output = self.nlin_layer(filtered_x)
        filt_rxa = self.rxa_filter_layers(nonlin_output)
        return filt_rxa

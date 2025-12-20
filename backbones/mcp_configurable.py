import torch.nn as nn

from backbones.modules_filter import (
    TxaFilterEnsembleTorch,
    RxaFilterEnsembleTorch,
)

from backbones.modules_mlp import SingleLayerPerceptron


class NlinCore(nn.Module):
    def __init__(self, n_channels, nonlinearity="silu"):
        super().__init__()
        self.n_channels = n_channels
        layers = []
        layers.append(
            SingleLayerPerceptron(
                n_channels, 32, 32)
        )
        layers.append(
            SingleLayerPerceptron(
                n_channels, 32, 32)
        )
        layers.append(
            SingleLayerPerceptron(
                n_channels, 32, 32)
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        batch, time = x.shape[0], x.shape[1]
        x_flat = x.view(batch * time, -1)  # Shape: (B*T, C*2)
        x_flat = self.model(x_flat)
        x_flat = x_flat.view(batch, time, self.n_channels, 2)
        return x_flat


class MCPConfig(nn.Module):
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

    def forward(self, x, h_0=None):
        filtered_x = self.txa_filter_layers(x)
        nonlin_output = self.nlin_layer(filtered_x)
        filt_rxa = self.rxa_filter_layers(nonlin_output)
        return filt_rxa

import torch.nn as nn

from backbones.modules_filter import (
    TxaFilterEnsembleTorch,
    RxaFilterEnsembleTorch,
)

from backbones.modules_mlp import SingleLayerPerceptron


class NlinCore(nn.Module):
    def __init__(self, n_channels, num_layers, nonlinearity="gelu"):
        super().__init__()
        self.n_channels = n_channels
        layers = []
        for _ in range(num_layers):
            layers.append(
                SingleLayerPerceptron(
                    n_channels, nonlinearity,
                    2 * n_channels,
                    2 * n_channels
                )
            )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch, time, n_channels, 2)
        batch_size, seq_len = x.shape[0], x.shape[1]
        # Flatten for processing: (batch * seq_len, C*2)
        x_flat = x.view(batch_size * seq_len, -1)
        transformed = self.model(x_flat)
        # Reshape back: (batch, time, n_channels, 2)
        transformed = transformed.view(batch_size, seq_len, self.n_channels, 2)
        return transformed


class MultiChannelMLP(nn.Module):
    def __init__(self, seq_len, tx_filt_size, rx_filt_size, n_channels):
        super().__init__()
        num_mlp_layers = 2

        self.txa_filter_layers = TxaFilterEnsembleTorch(
            n_channels, tx_filt_size, seq_len
        )

        self.nlin_layer = NlinCore(
            n_channels, num_mlp_layers
        )

        self.rxa_filter_layers = RxaFilterEnsembleTorch(
            n_channels, rx_filt_size, seq_len
        )

    def forward(self, x, h_0=None):
        filtered_x = self.txa_filter_layers(x)
        nonlin_output = self.nlin_layer(filtered_x)
        filt_rxa = self.rxa_filter_layers(nonlin_output)
        return filt_rxa

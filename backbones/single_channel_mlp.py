import torch
import torch.nn as nn

from backbones.common_modules import (
    TxaFilterEnsembleTorch,
    RxaFilterEnsembleTorch,
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


class NlinCore(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.nonlin = PhaseAwareNonlin()

    def forward(self, x):
        return self.nonlin(x)


class SingleChannelMLP(nn.Module):
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

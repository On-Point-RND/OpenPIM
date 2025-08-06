import torch
import torch.nn as nn

from backbones.common_modules import (
    TxaFilterEnsembleTorch, RxaFilterEnsembleTorch
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
        self.pre_nlin_mix = nn.Linear(
            2 * n_channels,
            2,
            bias=False
        )
        self.nlin = PhaseAwareNonlin()

    def forward(self, x, h_0=None):
        b, s, c, _ = x.shape
        mixed_x = self.pre_nlin_mix(
            x.reshape(b * s, c * 2)
        )
        nlin_distorted = self.nlin(mixed_x)
        # Reshape back to (batch, time_seq, 2)
        nlin_distorted = nlin_distorted.view(b, s, 2)
        # Copy 16 times to create (batch, time_seq, 16, 2)
        nlin_distorted = nlin_distorted.unsqueeze(2).expand(-1, -1, 16, -1)
        return nlin_distorted


class ExternalSingle(nn.Module):
    def __init__(self, in_seq_size, out_seq_size, n_channels):
        super().__init__()
        self.in_seq_size = in_seq_size
        self.out_seq_size = out_seq_size
        self.n_channels = n_channels

        self.txa_filter_layers = TxaFilterEnsembleTorch(
            n_channels, in_seq_size, out_seq_size
        )

        self.nlin_core = NlinCore(
            n_channels
        )

        self.rxa_filter_layers = RxaFilterEnsembleTorch(
            n_channels, out_seq_size
        )

        self.bn_output = nn.BatchNorm1d(n_channels)

    def forward(self, x, h_0=None):
        filtered_x = self.txa_filter_layers(x)
        nonlin_output = self.nlin_core(filtered_x)
        output = self.rxa_filter_layers(nonlin_output)
        output = self.bn_output(output)
        return output

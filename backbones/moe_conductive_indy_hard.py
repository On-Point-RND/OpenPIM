import torch
import torch.nn as nn

from backbones.common_modules import (
    TxaFilterEnsembleTorch,
    RxaFilterEnsembleTorch,
)


class MoELayer(nn.Module):
    def __init__(self, n_seq):
        super().__init__()
        self.n_seq = n_seq
        experts_nlin = [
            "relu", "silu", "none"
        ]
        self.experts = nn.ModuleList([
            PhaseAwareNonlin(nonlinearity=expert)
            for expert in experts_nlin
        ])

    def forward(self, x):
        experts_outputs = torch.stack(
            [expert(x) for expert in self.experts]
        )
        # Average the expert outputs with equal weights of 1/3
        weighted_output = torch.mean(experts_outputs, dim=0)
        return weighted_output


class PhaseAwareNonlin(nn.Module):
    def __init__(self, hidden_size=16, num_layers=2, nonlinearity="silu"):
        super().__init__()
        layers = []
        # Set non-linearity
        self.nlin = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "silu": nn.SiLU(),
            "none": nn.Identity(),
            "gelu": nn.GELU(),
            "selu": nn.SELU(),
            "softplus": nn.Softplus(),
        }[nonlinearity]
        # Input: [I, Q, |x|] (3 features)
        for i in range(num_layers):
            in_dim = 3 if i == 0 else hidden_size
            out_dim = 2 if i == num_layers - 1 else hidden_size
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(self.nlin)
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


class MoEConductiveIndyHard(nn.Module):
    def __init__(self, in_seq_size, out_seq_size, n_channels):
        super().__init__()
        self.out_seq_size = out_seq_size
        self.n_channels = n_channels
        self.moe_layer = MoELayer(out_seq_size)
        self.txa_filter_layers = TxaFilterEnsembleTorch(
            n_channels, in_seq_size, out_seq_size
        )

        self.rxa_filter_layers = RxaFilterEnsembleTorch(n_channels, out_seq_size)

        self.bn_output = nn.BatchNorm1d(n_channels)  # For complex output

    def forward(self, x, h_0=None):
        filtered_x = self.txa_filter_layers(x)
        weighted_output = self.moe_layer(filtered_x)
        filt_rxa = self.rxa_filter_layers(weighted_output)
        output = self.bn_output(filt_rxa)
        return output

import torch
import torch.nn as nn
import torch.nn.init as init

from backbones.common_modules import (
    TxaFilterEnsembleTorch,
    RxaFilterEnsembleTorch,
)


class PhaseAwareNonlin(nn.Module):
    def __init__(self, hidden_size=16, preproc_dim=3, num_layers=2):
        super().__init__()
        layers = []
        # Input: [I, Q, |x|] (3 features)
        for i in range(num_layers):
            in_dim = 3 if i == 0 else hidden_size
            out_dim = preproc_dim if i == num_layers - 1 else hidden_size
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


class SingleLayerPerceptron(nn.Module):
    def __init__(self, n_channels, nonlinearity, input_size=32):
        super().__init__()
        self.n_channels = n_channels
        self.linear = nn.Linear(input_size, 2 * n_channels, bias=True)
        self._initialize_as_identity()

        # Set non-linearity
        self.nlin = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "silu": nn.SiLU(),
            "gelu": nn.GELU(),
            "none": nn.Identity(),
        }[nonlinearity]

    def _initialize_as_identity(self):
        init.eye_(self.linear.weight)
        # Optional: zero out the bias
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, x):
        batch, time = x.shape[0], x.shape[1]
        x_flat = x.view(batch * time, -1)
        transformed = self.linear(x_flat)
        transformed = transformed.view(batch, time, self.n_channels, 2)
        return self.nlin(transformed)


class NlinCore(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        preproc_dim = 3
        preproc_layers = 2
        nonlinearity = "silu"
        num_layers = 3
        layers = []
        self.preproc_layer = PhaseAwareNonlin(
            preproc_dim=preproc_dim,
            num_layers=preproc_layers
        )
        for i in range(num_layers):
            layers.append(
                SingleLayerPerceptron(
                    n_channels,
                    nonlinearity,
                    input_size=preproc_dim*n_channels if i == 0 else 32
                )
            )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        preproc_x = self.preproc_layer(x)
        return self.model(preproc_x)


class McMLPPreproc(nn.Module):
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

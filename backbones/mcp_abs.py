import torch
import torch.nn as nn
import torch.nn.init as init

from backbones.filter_modules import (
    TxaFilterEnsembleTorch,
    RxaFilterEnsembleTorch,
)

from backbones.mlp_modules import (
    SingleLayerPerceptron,
)

class EnrichedPerceptron(nn.Module):
    def __init__(self, n_channels, nonlinearity):
        super().__init__()
        self.n_channels = n_channels
        self.linear = nn.Linear(3 * n_channels, 2 * n_channels, bias=True)
        self._initialize_as_identity()

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
        # x is expected to be (batch * time, n_ch, 2)
        batch_time, n_ch, _ = x.shape
        # Extract real and imaginary parts
        x_real = x[..., 0]  # Shape: (B*T, C)
        x_imag = x[..., 1]  # Shape: (B*T, C)

        # Calculate modulus square: |x|² = real² + imag²
        modulus_square = x_real.pow(2) + x_imag.pow(2)  # Shape: (B*T, C)

        # Concatenate: [real, imag, |x|²]
        x_expanded = torch.cat([x_real, x_imag, modulus_square], dim=-1)
        # Model acts on shapes: (B*T, C*3) -> (B*T, C*2)
        transformed = self.linear(x_expanded)
        transformed = transformed.view(batch_time, n_ch, 2)
        return self.nlin(transformed)


class NlinCore(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        nonlinearity = "silu"
        num_layers = 3
        layers = []
        layers.append(EnrichedPerceptron(n_channels, nonlinearity))
        for _ in range(num_layers - 1):
            layers.append(SingleLayerPerceptron(
                n_channels, 
                nonlinearity,
                input_size=2 * n_channels,
                output_size=2 * n_channels
            ))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        batch, time_seq_len = x.shape[0], x.shape[1]
        n_ch = self.n_channels
        # Flatten for processing: (batch * time_seq_len, n_ch, 2)
        x_flat = x.view(batch * time_seq_len, n_ch, 2)
        transformed = self.model(x_flat)
        # Reshape back: (batch, time_seq_len, n_channels, 2)
        transformed = transformed.view(batch, time_seq_len, n_ch, 2)
        return transformed


class McpAbs(nn.Module):
    def __init__(self, in_seq_size, out_seq_size, n_channels):
        super().__init__()
        self.n_channels = n_channels

        self.txa_filter_layers = TxaFilterEnsembleTorch(
            n_channels, in_seq_size, out_seq_size
        )

        self.nlin_layer = NlinCore(
            n_channels,
        )

        self.rxa_filter_layers = RxaFilterEnsembleTorch(
            n_channels, out_seq_size
        )

    def forward(self, x, h_0=None):
        filtered_x = self.txa_filter_layers(x)
        nonlin_output = self.nlin_layer(filtered_x)
        filt_rxa = self.rxa_filter_layers(nonlin_output)
        return filt_rxa

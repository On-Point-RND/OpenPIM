import torch
import torch.nn as nn


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.fc_real = nn.Linear(in_features, out_features, bias=bias)
        self.fc_imag = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x_real, x_imag):
        real = self.fc_real(x_real) - self.fc_imag(x_imag)
        imag = self.fc_real(x_imag) + self.fc_imag(x_real)
        return real, imag


class PolynomialExpansion(nn.Module):
    def __init__(self, max_degree):
        super().__init__()
        self.max_degree = max_degree

    def forward(self, x_real, x_imag):
        x = torch.complex(x_real, x_imag)
        outputs_real = []
        outputs_imag = []
        for d in range(1, self.max_degree + 1):
            x_pow = x**d
            outputs_real.append(x_pow.real)
            outputs_imag.append(x_pow.imag)
        return torch.cat(outputs_real, dim=-1), torch.cat(outputs_imag, dim=-1)


class Linear(nn.Module):
    def __init__(self, input_size, output_size, n_channels, batch_size, poly_degree=3):
        super().__init__()
        self.input_size = input_size
        self.n_channels = n_channels
        self.poly_degree = poly_degree

        # Corrected batch normalization for expanded features
        self.bn_input = nn.BatchNorm1d(n_channels * 2 * poly_degree)
        self.bn_output = nn.BatchNorm1d(n_channels * 2)

        # Initialize ComplexLinear with correct input_size
        self.complex_fc_in = ComplexLinear(input_size * 16, input_size)
        self.poly_expand = PolynomialExpansion(poly_degree)
        expanded_size = input_size * poly_degree  # Compute expanded size
        self.complex_fc = ComplexLinear(expanded_size, output_size // 2)
        self.cross_channel_fc = ComplexLinear(
            n_channels * (output_size // 2), output_size // 2
        )

    def forward(self, x, h_0=None):
        B, N, C, _ = x.shape  # Input shape: (B, N, C, 2)

        # Flatten N and C dimensions for processing
        x_real = (
            x[..., 0].permute(0, 2, 1).contiguous().view(B, -1)
        )  # Shape: (B, C * N)
        x_imag = (
            x[..., 1].permute(0, 2, 1).contiguous().view(B, -1)
        )  # Shape: (B, C * N)

        # Initial complex processing
        x_real, x_imag = self.complex_fc_in(x_real, x_imag)

        # Polynomial expansion
        x_real, x_imag = self.poly_expand(
            x_real, x_imag
        )  # Output shape: (B, expanded_size)

        # Further processing
        x_real, x_imag = self.complex_fc(x_real, x_imag)

        # Cross-channel processing
        x_real = (
            x_real.view(B, self.n_channels, -1)
            .permute(0, 2, 1)
            .contiguous()
            .view(B, -1)
        )
        x_imag = (
            x_imag.view(B, self.n_channels, -1)
            .permute(0, 2, 1)
            .contiguous()
            .view(B, -1)
        )
        x_real, x_imag = self.cross_channel_fc(x_real, x_imag)

        # Combine real/imaginary parts
        output = torch.stack([x_real, x_imag], dim=-1).view(B, self.n_channels, 2)
        output = self.bn_output(output.view(B, -1)).view(B, self.n_channels, 2)
        return output

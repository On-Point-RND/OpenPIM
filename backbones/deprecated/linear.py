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


class Linear(nn.Module):
    def __init__(self, input_size, output_size, n_channels, batch_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_channels = n_channels

        # Batch normalization layers
        self.bn_input = nn.BatchNorm1d(n_channels * 2)  # For complex input
        self.bn_output = nn.BatchNorm1d(n_channels * 2)  # For complex output

        # Complex linear layers
        self.complex_fc_in = ComplexLinear(input_size, input_size)
        self.complex_fc = ComplexLinear(input_size, output_size // 2)

    def forward(self, x, h_0=None):
        # INFO: input tensors of sizes B x N x C x 2
        B, N, C, _ = x.shape

        # INFO: resulting tensors of sizes B x C
        # Initialize output tensors
        final_real = torch.zeros((B, self.n_channels), device=x.device)
        final_imag = torch.zeros_like(final_real)

        for c in range(self.n_channels):
            # Extract real/imag components for current channel
            x_real = x[:, :, c, 0]  # (B, N)
            x_imag = x[:, :, c, 1]  # (B, N)

            # Initial complex processing
            x_real, x_imag = self.complex_fc_in(x_real, x_imag)

            # Amplitude modulation
            amp = x[:, :, c, 0].pow(2) + x[:, :, c, 1].pow(2)  # (B, N)
            x_real = amp * x_real
            x_imag = amp * x_imag

            # Final complex processing
            x_real, x_imag = self.complex_fc(x_real, x_imag)

            # Aggregate results (squeeze last dimension)
            final_real[:, c] = x_real.squeeze(-1)  # (B,)
            final_imag[:, c] = x_imag.squeeze(-1)  # (B,)

        # Combine real/imag parts and apply output normalization
        output = torch.stack([final_real, final_imag], dim=-1)  # (B, C, 2)
        output = output.view(B, -1)  # (B, 2C)
        output = self.bn_output(output)  # Normalize
        output = output.view(B, C, 2)  # Back to (B, C, 2)
        # INFO: resulting tensors of sizes B x C x 2
        return output

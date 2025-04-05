import torch
from torch import nn


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.fc_real = nn.Linear(in_features, out_features, bias=bias)
        self.fc_imag = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x_real, x_imag):
        return self.fc_real(x_real) - self.fc_imag(x_imag), self.fc_real(
            x_imag
        ) + self.fc_imag(x_real)


class Linear(nn.Module):
    def __init__(self, input_size, output_size, batch_size, n_channels):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.n_channels = n_channels
        # Complex linear layers
        self.complex_fc_in = ComplexLinear(input_size, input_size)
        self.complex_fc = ComplexLinear(input_size, output_size // 2)

    def forward(self, x, h_0):

        for c in range(self.n_channels):
            # Initial complex processing
            x_i[:, :, c], x_q[:, :, c] = x[:, :, c, 1], x[:, :, c, 1]
            c_real[:, :, c], c_imag[:, :, c] = self.complex_fc_in(
                x_i[:, :, c], x_q[:, :, c]
            )

            # Amplitude modulation
            amp2 = torch.sum(x[:, :, c] ** 2, dim=-1, keepdim=False)
            x_i[:, :, c], x_q[:, :, c] = amp2 * c_real[:, :, c], amp2 * c_imag[:, :, c]

            # Final complex processing
            c_real[:, :, c], c_imag[:, :, c] = self.complex_fc(
                x_i[:, :, c], x_q[:, :, c]
            )
        return torch.cat([C_real, C_imag], dim=-1)

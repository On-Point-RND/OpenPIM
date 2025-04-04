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

        self.bn_in = nn.BatchNorm1d(num_features=2)
        self.bn_out = nn.BatchNorm1d(num_features=2)

        # Complex linear layers
        self.complex_fc_in = ComplexLinear(input_size, input_size)
        self.complex_fc = ComplexLinear(input_size, output_size // 2)

    def forward(self, x, h_0):
        # Input processing
        x = x.permute(0, 2, 1)
        x = self.bn_in(x)
        x = x.permute(0, 2, 1)

        # Initial complex processing
        x_i, x_q = x[..., 0], x[..., 1]
        c_real, c_imag = self.complex_fc_in(x_i, x_q)

        # Amplitude modulation
        amp2 = torch.sum(x**2, dim=-1, keepdim=False)
        x_i, x_q = amp2 * c_real, amp2 * c_imag

        # Final complex processing
        C_real, C_imag = self.complex_fc(x_i, x_q)
        return self.bn_out(torch.cat([C_real, C_imag], dim=-1))

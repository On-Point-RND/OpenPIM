import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, input_size, output_size, batch_size):
        super(Linear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size

        self.bn_in = nn.BatchNorm1d(num_features=2)
        self.bn_out = nn.BatchNorm1d(num_features=2)


        # Split complex weights into real and imaginary components
        self.fc_I = nn.Linear(in_features=self.input_size,
                              out_features=self.output_size//2,
                              bias=False)
        self.fc_Q = nn.Linear(in_features=self.input_size,
                              out_features=self.output_size//2,
                              bias=False)

    def forward(self, x, h_0):
        # # Input processing
        # x = x.permute(0, 2, 1)  # [batch, channels, time]
        # x = self.bn_in(x)
        # x = x.permute(0, 2, 1)  # [batch, time, channels]

        # Amplitude calculation
        # amp2 = (torch.pow(x[:, :, 0], 2) + torch.pow(x[:, :, 1], 2))
        # amp2 = amp2.unsqueeze(-1)
        # x = amp2 * x

        # Split into real and imaginary components
        x_i = x[:, :, 0]  # Real part
        x_q = x[:, :, 1]  # Imaginary part

        # Complex matrix multiplication decomposition
        real_real = self.fc_I(x_i)  # W_real @ x_real
        real_imag = self.fc_I(x_q)  # W_real @ x_imag
        imag_real = self.fc_Q(x_i)  # W_imag @ x_real
        imag_imag = self.fc_Q(x_q)  # W_imag @ x_imag

        # Combine results
        C_real = real_real - imag_imag
        C_imag = real_imag + imag_real

        # Concatenate and apply output batch norm
        out = torch.cat([C_real, C_imag], dim=-1)
        return  out #self.bn_out(out)



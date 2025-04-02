import torch
from torch import nn


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.fc_real = nn.Linear(in_features, out_features, bias=bias)
        self.fc_imag = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x_real, x_imag):
        return self.fc_real(x_real) - self.fc_imag(x_imag), self.fc_real(x_imag) + self.fc_imag(x_real)

class Linear(nn.Module):
    def __init__(self, n_channels, input_size, output_size, batch_size):
        super().__init__()
        self.n_channels = n_channels 
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size

        self.bn_in = nn.BatchNorm1d(num_features=n_channels*2)
        self.bn_out = nn.BatchNorm1d(num_features=2)
        
        # Complex linear layers
        self.complex_fc_in_s = nn.ModuleList()
        for i in range(0,n_channels):
            layer = ComplexLinear(input_size, input_size)
            self.complex_fc_in_s.append(layer)
            
        self.complex_fc = ComplexLinear(input_size*n_channels, output_size//2)

    def forward(self, x, h_0):
        # Input processing
        x = x.permute(0, 2, 1)
        # print('x.shape: ', x.shape)
        x = self.bn_in(x)
        x = x.permute(0, 2, 1)

        # Initial complex processing
        # x_i, x_q = x[..., 0], x[..., 1]
        
        for id,complex_layer in enumerate(self.complex_fc_in_s):
            x_i, x_q = x[..., 2*id], x[..., 2*id+1]  
            c_real, c_imag = complex_layer(x_i, x_q)
            amp2 = x[..., 2*id]**2 + x[..., 2*id+1]**2
            x_i, x_q = amp2 * c_real, amp2 * c_imag

            if id ==0:
                x_ii = x_i
                x_qq = x_q
            else:
                x_ii = torch.cat([x_ii, x_i], dim=-1)
                x_qq = torch.cat([x_qq, x_q], dim=-1)
            
        # Amplitude modulation

        # Final complex processing
        C_real, C_imag = self.complex_fc(x_ii, x_qq)
        return self.bn_out(torch.cat([C_real, C_imag], dim=-1))



from torch import nn
import torch

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.fc_real = nn.Linear(in_features, out_features, bias=bias)
        self.fc_imag = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x_real, x_imag):
        return self.fc_real(x_real) - self.fc_imag(x_imag), self.fc_real(x_imag) + self.fc_imag(x_real)

class ComplexRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, batch_first, bias):
        super().__init__()
        
        self.rnn_real = nn.LSTM(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          batch_first=batch_first,
                          bias=bias)
        
        self.rnn_imag = nn.LSTM(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          batch_first=batch_first,
                          bias=bias)

    def forward(self, x_real, x_imag):
        real_r, (_,_) = self.rnn_real(x_real)
        imag_i, (_,_) = self.rnn_imag(x_imag)
        real_i, (_,_) = self.rnn_real(x_imag)
        imag_r, (_,_) = self.rnn_imag(x_real)

        return real_r - imag_i, real_i + imag_r


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_size, bidirectional=False, batch_first=True,
                 bias=True):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size=batch_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.bias = bias

        # Instantiate NN Layers
        self.bn_in = nn.BatchNorm1d(num_features=2)
        self.bn_out = nn.BatchNorm1d(num_features=2)

        self.complex_fc = ComplexLinear(input_size, output_size//2)
                
        self.complex_fc_in = ComplexRNN(input_size=input_size,
                          hidden_size=input_size,
                          num_layers=num_layers,
                          bidirectional=self.bidirectional,
                          batch_first=self.batch_first,
                          bias=self.bias)

    def forward(self, x, h_0):

        # print('x.shape: ', x.shape)
        
        x = x.permute(0, 2, 1)
        x = self.bn_in(x)
        x = x.permute(0, 2, 1)

        # Initial complex processing
        x_i, x_q = x[..., 0], x[..., 1]
        c_real, c_imag = self.complex_fc_in(x_i, x_q)

        # Amplitude modulation
        amp2 = torch.sum(x**2, dim=-1, keepdim=False)
        x_i, x_q = amp2 * c_real, amp2 * c_imag

        # x_i, x_q = self.rnn(x_i, x_q)

        # Final complex processing
        C_real, C_imag = self.complex_fc(x_i, x_q)
        return self.bn_out(torch.cat([C_real, C_imag], dim=-1))

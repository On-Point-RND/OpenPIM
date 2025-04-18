"""
Description: Quantized GRU (QGRU) backbone
"""

import torch
from torch import nn
from quant import Sqrt, Pow

class QGRU(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, batch_size, bidirectional=False, batch_first=True,
                 bias=True):
        super(QGRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = 4
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size=batch_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.bias = bias

        # Instantiate NN Layers
        self.bn_in = nn.InstanceNorm2d(num_features=self.batch_size)
        self.bn_out = nn.InstanceNorm2d(num_features=self.batch_size)
        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=self.bidirectional,
                          batch_first=self.batch_first,
                          bias=self.bias)

        
        self.fc_out = nn.Linear(in_features=hidden_size,
                                out_features=self.output_size,
                                bias=self.bias)

        
    def reset_parameters(self):
        for name, param in self.rnn.named_parameters():
            num_gates = int(param.shape[0] / self.hidden_size)
            if 'bias' in name:
                nn.init.constant_(param, 0)
            if 'weight' in name:
                for i in range(0, num_gates):
                    nn.init.orthogonal_(param[i * self.hidden_size:(i + 1) * self.hidden_size, :])
            if 'weight_ih_l0' in name:
                for i in range(0, num_gates):
                    nn.init.xavier_uniform_(param[i * self.hidden_size:(i + 1) * self.hidden_size, :])

        for name, param in self.fc_out.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            if 'bias' in name:
                nn.init.constant_(param, 0)

        for name, param in self.fc_hid.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_uniform_(param)
            if 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x, h_0):
        # Feature Extraction
        i_x = torch.unsqueeze(x[..., 0], dim=-1)
        q_x = torch.unsqueeze(x[..., 1], dim=-1)
        amp2 = torch.pow(i_x, 2) + torch.pow(q_x, 2)
        # amp2 = self.pow2(i_x) + self.pow2(q_x)
        amp = torch.sqrt(amp2)
        # amp = self.sqrt(amp2)
        amp3 = torch.pow(amp, 3)
        # amp3 = self.pow3(amp)
        
        x = torch.cat((i_x, q_x, amp, amp3), dim=-1)


        # Regressor
        x = self.bn_in(x)
        out, _ = self.rnn(x, h_0)
        out = self.fc_out(out)
        out = self.bn_out(out)
        return out

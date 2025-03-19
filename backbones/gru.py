from torch import nn
import torch


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_size, bidirectional=False, batch_first=True,
                 bias=True):
        super(GRU, self).__init__()
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
        self.rnn_I = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=self.bidirectional,
                          batch_first=self.batch_first,
                          bias=self.bias)
        self.rnn_Q = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=self.bidirectional,
                          batch_first=self.batch_first,
                          bias=self.bias)

        self.fc_I = nn.Linear(in_features=hidden_size,
                                out_features=self.output_size//2,
                                bias=True)
        
        self.fc_Q = nn.Linear(in_features=hidden_size,
                                out_features=self.output_size//2,
                                bias=True)

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

    def forward(self, x, h_0):

        # print('x.shape: ', x.shape)
        
        x = x.permute(0, 2, 1)
        x = self.bn_in(x)
        x = x.permute(0, 2, 1)

        amp2 = torch.pow(x[:, :, 0], 2) + torch.pow(x[:, :, 1], 2)  # Shape: (batch_size, sequence_length)
        amp2 = amp2.unsqueeze(-1)  # Shape: (batch_size, sequence_length, 1)
        x = amp2 * x  # Shape: (batch_size, sequence_length, input_size)

        # Split the input into I and Q components
        x_i = x[:, :, 0]  # Shape: (batch_size, sequence_length)
        x_q = x[:, :, 1]  # Shape: (batch_size, sequence_length)

        
        # print('x_i.shape: ', x_i.shape)
        # x_i, (_, _) = self.rnn_I(x_i, (h_0, h_0))
        # x_q, (_, _) = self.rnn_Q(x_q, (h_0, h_0))

        x_i, _ = self.rnn_I(x_i)
        x_q, _ = self.rnn_Q(x_q)
        
        x_i = self.fc_I(x_i)
        x_q = self.fc_Q(x_q)
        
        out = torch.cat([x_i, x_q], dim=-1)  # Shape: (batch_size, sequence_length, output_size)
        # print('out.shape: ', out.shape)
        return self.bn_out(out)

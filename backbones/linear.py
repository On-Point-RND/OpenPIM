from torch import nn
import torch

class Linear(nn.Module):
    def __init__(self, input_size, output_size, batch_size):
        super(Linear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size=batch_size

        # Instantiate NN Layers
        # self.bn_in = nn.InstanceNorm1d(num_features=self.batch_size)
        # self.bn_out = nn.InstanceNorm1d(num_features=self.batch_size)
        self.bn_in = nn.BatchNorm1d(num_features=2)
        self.bn_out = nn.BatchNorm1d(num_features=2)

        self.fc_I = nn.Linear(in_features=self.input_size,
                                out_features=self.output_size//2,
                                bias=False)
        
        self.fc_Q = nn.Linear(in_features=self.input_size,
                                out_features=self.output_size//2,
                                bias=False)
                                
    
    def forward(self, x, h_0):
        # Compute amplitude squared
        x = x.permute(0, 2, 1)
        x = self.bn_in(x)
        x = x.permute(0, 2, 1)

        ri, rq = x[:, -2, 0], x[:, -2, 1]

        
        amp2 = torch.pow(x[:, :, 0], 2) + torch.pow(x[:, :, 1], 2)  # Shape: (batch_size, sequence_length)
        amp2 = amp2.unsqueeze(-1)  # Shape: (batch_size, sequence_length, 1)
        x = amp2 * x  # Shape: (batch_size, sequence_length, input_size)

        # Split the input into I and Q components
        x_i = x[:, :, 0]  # Shape: (batch_size, sequence_length)
        x_q = x[:, :, 1]  # Shape: (batch_size, sequence_length)
        
        # Pass through the linear layers
        x_i = self.fc_I(x_i)  # Shape: (batch_size, sequence_length, output_size // 2)
        x_q = self.fc_Q(x_q)  # Shape: (batch_size, sequence_length, output_size // 2)
        
        # Concatenate I and Q components along the last dimension
        out = torch.cat([x_i, x_q], dim=-1)  # Shape: (batch_size, sequence_length, output_size)
        #print(out.shape, torch.cat([x[:, -1, 0], x[:, -1, 1]], dim=-1).shape)
        return self.bn_in(out)



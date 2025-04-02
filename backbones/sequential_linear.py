import torch
import torch.nn as nn
import torch.nn.functional as F

# Existing ComplexLinear class
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.fc_real = nn.Linear(in_features, out_features, bias=bias)
        self.fc_imag = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x_real, x_imag):
        return (
            self.fc_real(x_real) - self.fc_imag(x_imag),
            self.fc_real(x_imag) + self.fc_imag(x_real),
        )

# Existing Linear class
class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
      

        # Complex linear layers
        self.complex_fc_in = ComplexLinear(input_size, input_size)
        self.complex_fc = ComplexLinear(input_size, output_size)

    def forward(self, x):
       

        # Initial complex processing
        x_i, x_q = x[..., 0], x[..., 1]
        c_real, c_imag = self.complex_fc_in(x_i, x_q)

        # Amplitude modulation
        amp2 = torch.sum(x**2, dim=-1, keepdim=False)
        x_i, x_q = amp2 * c_real, amp2 * c_imag

        # Final complex processing
        C_real, C_imag = self.complex_fc(x_i, x_q)
        return torch.cat([C_real, C_imag], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, skip=True):
        super().__init__()
        self.linear1 = ComplexLinear(input_size, output_size)
        self.relu = nn.ReLU()
        self.linear2 = ComplexLinear(output_size, output_size)
        self.skip = skip

    def forward(self, x):
        # First Linear layer
        x_i, x_q = x[..., 0], x[..., 1]
        x_i, x_q  = self.linear1(x_i, x_q)
        
        x_i, x_q  = self.relu(x_i), self.relu(x_q)
        # Second Linear layer
        x_i, x_q = self.linear2(x_i, x_q)
        # Skip connection
        if self.skip:
            x_i, x_q = x[..., 0]+x_i, x[..., 1]+x_q  # Add the input to the output (skip connection)
        return torch.cat([x_i.unsqueeze(-1), x_q.unsqueeze(-1)], dim=-1)

# Sequential Model with Repeated Blocks
class SequentialLinear(nn.Module):
    def __init__(self, input_size,  output_size, batch_size, hidden_size=256, num_blocks=10):
        super().__init__()
        bloks = [ResidualBlock(input_size, hidden_size, skip=False)]
        for _ in range(num_blocks-1):
            bloks.append(ResidualBlock(hidden_size, hidden_size))
        self.blocks = nn.ModuleList(bloks)
        self.output_layer = Linear(hidden_size, output_size//2)
    def forward(self, x, h_0):
        # Initial transformation
        # Pass through residual blocks
        for block in self.blocks:
            x = block(x)
        # Final transformationx
        #x = x.view(batch_size)
        x = self.output_layer(x)
        return x


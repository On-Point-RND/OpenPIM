import torch
import torch.nn as nn

from backbones.common_modules import (
    TxaFilterEnsemble, RxaFilterEnsemble
)


class LinearLeakageNlinCore(nn.Module):
    def __init__(self, seq_len, output_size, n_channels):
        super().__init__()
        self.seq_len = seq_len
        self.output_size = output_size
        self.n_channels = n_channels

        self.nonlin_layers = nn.ModuleList()
        for _ in range(n_channels):
            layer = nn.Linear(output_size * (n_channels - 1), output_size, bias=False)
            self.nonlin_layers.append(layer)

    def forward(self, x):
        b, s, c, _ = x.shape
        x_reshaped = x.reshape(b * s, c, 2)
        
        # Calculate nonlinearity coefficient
        amp = x_reshaped[..., 0].pow(2) + x_reshaped[..., 1].pow(2)
        
        # Apply nonlinearity to distort signals
        nlin_distorted = amp.unsqueeze(-1) * x_reshaped
        
        # Reshape to combine batch and sequence dimensions
        nlin_distorted = nlin_distorted.reshape(b, s, c, 2)

        # Split complex input into real and imaginary parts
        filtered_real = nlin_distorted[..., 0]  # shape: (B, S, C)
        filtered_imag = nlin_distorted[..., 1]  # shape: (B, S, C)
        
        # Initialize output tensor
        nonlin_output = torch.empty_like(x)

        # Create a mask for all channels except current
        channel_mask = ~torch.eye(self.n_channels, dtype=torch.bool, device=x.device)

        for id in range(self.seq_len):
            for c, nonlin_layer in enumerate(self.nonlin_layers):
                # Get all channels except current using the mask
                filtered_real_c = filtered_real[:, id][:, channel_mask[c]]  # shape: (B, C-1)
                filtered_imag_c = filtered_imag[:, id][:, channel_mask[c]]  # shape: (B, C-1)
                
                # Concatenate real and imaginary parts
                total_filtered = torch.cat([filtered_real_c, filtered_imag_c], dim=-1)  # shape: (B, 2*(C-1))
                
                # Apply nonlinear layer
                c_output = nonlin_layer(total_filtered)  # shape: (B, 2)
                nonlin_output[:, id, c, 0] = c_output[:, 0]
                nonlin_output[:, id, c, 1] = c_output[:, 1]
        return nonlin_output


class LinearLeakage(nn.Module):
    def __init__(self, input_size, output_size,
                 n_channels, batch_size, out_window=10):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_channels = n_channels
        self.out_window = out_window

        self.txa_filter_layers = TxaFilterEnsemble(
            n_channels, input_size, out_window
        )

        self.nlin_core = LinearLeakageNlinCore(
            out_window, output_size, n_channels
        )

        self.rxa_filter_layers = RxaFilterEnsemble(
            n_channels, out_window
        )

        # Batch normalization layers
        self.bn_output = nn.BatchNorm1d(n_channels)  # For complex output

    def forward(self, x, h_0=None):
        filtered_x = self.txa_filter_layers(x)
        nonlin_output = self.nlin_core(filtered_x)
        output = self.rxa_filter_layers(nonlin_output)
        output = self.bn_output(output)
        return output

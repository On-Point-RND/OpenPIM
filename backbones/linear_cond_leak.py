import torch.nn as nn

from backbones.common_modules import (
    TxaFilterEnsemble, RxaFilterEnsemble
)


class LinearCLCore(nn.Module):
    def __init__(self, seq_len, output_size, n_channels):
        super().__init__()
        self.seq_len = seq_len
        self.output_size = output_size
        self.n_channels = n_channels
        self.post_nlin_mix = nn.Linear(
            output_size * n_channels,
            output_size * n_channels,
            bias=False
        )

    def forward(self, x, h_0=None):
        # Reshape to combine batch and sequence dimensions
        b, s, c, _ = x.shape
        
        # Calculate nonlinearity coefficient
        amp = x[..., 0].pow(2) + x[..., 1].pow(2)
        
        # Apply nonlinearity to distort signals
        nlin_distorted = amp.unsqueeze(-1) * x
        
        # Reshape to combine batch and sequence,
        # channel and complex dimensions, and apply nonlinearity
        nlin_distorted = nlin_distorted.reshape(b * s, c * 2)
        
        output = self.post_nlin_mix(nlin_distorted)
        return output.reshape(b, s, c, 2)


class LinearCondLeak(nn.Module):
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

        self.nonlin_core = LinearCLCore(
            out_window, output_size, n_channels
        )

        self.rxa_filter_layers = RxaFilterEnsemble(
            n_channels, out_window
        )

        self.bn_output = nn.BatchNorm1d(n_channels)  # For complex output

    def forward(self, x, h_0=None):
        filtered_x = self.txa_filter_layers(x)
        nonlin_output = self.nonlin_core(filtered_x)
        output = self.rxa_filter_layers(nonlin_output)
        output = self.bn_output(output)
        return output

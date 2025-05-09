import torch.nn as nn

from backbones.common_modules import (
    TxaFilterEnsemble, RxaFilterEnsemble
)


class LinearExternalNlinCore(nn.Module):
    def __init__(self, seq_len, output_size, n_channels):
        super().__init__()
        self.seq_len = seq_len
        self.output_size = output_size
        self.n_channels = n_channels
        self.pre_nlin_mix = nn.Linear(
            output_size * n_channels,
            output_size * n_channels,
            bias=False
        )
        self.coeff = nn.Linear(output_size, output_size, bias=False)

    def forward(self, x, h_0=None):
        b, s, c, _ = x.shape
        mixed_x = self.pre_nlin_mix(
            x.reshape(b * s, c * 2)
        )
        mixed_x = mixed_x.reshape(b * s, c, 2)
        # Calculate nonlinearity coefficient
        amp = mixed_x[..., 0].pow(2) + mixed_x[..., 1].pow(2)
        amp = amp.unsqueeze(-1)
        # Apply nonlinearity to already mixed signals to distort them
        nlin_distorted = amp * mixed_x
        output = self.coeff(
            nlin_distorted.reshape(b * s * c, 2)
        )
        return output.reshape(b, s, c, 2)


class LinearExternal(nn.Module):
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

        self.nlin_core = LinearExternalNlinCore(
            out_window, output_size, n_channels
        )

        self.rxa_filter_layers = RxaFilterEnsemble(
            n_channels, out_window
        )

        self.bn_output = nn.BatchNorm1d(n_channels)

    def forward(self, x, h_0=None):
        filtered_x = self.txa_filter_layers(x)
        nonlin_output = self.nlin_core(filtered_x)
        output = self.rxa_filter_layers(nonlin_output)
        output = self.bn_output(output)
        return output

import torch
import torch.nn as nn
from backbones.common_modules import RxaFilterEnsemble, TxaFilterEnsemble


class LinearInternalBody(nn.Module):
    def __init__(self, seq_len, output_size, n_channels):
        super().__init__()
        self.seq_len = seq_len
        self.output_size = output_size
        self.n_channels = n_channels

        self.nonlin_layers = nn.ModuleList()
        for _ in range(n_channels):
            layer = nn.Linear(output_size, output_size, bias=False)
            self.nonlin_layers.append(layer)

        self.coeff_layers = nn.ModuleList()
        for _ in range(n_channels):
            layer = nn.Linear(output_size, output_size, bias=False)
            self.coeff_layers.append(layer)

    def forward(self, x):
        nonlin_output = torch.empty_like(x, device=x.device)
        for id in range(self.seq_len):
            for c, nonlin_layer in enumerate(self.nonlin_layers):
                x_slice = nonlin_layer(x[:, id, c, :])
                c_output = torch.empty_like(x_slice)
                amp = x_slice[:, 0].pow(2) + x_slice[:, 1].pow(2)  # (B, N)
                c_output[:, 0] = amp * x_slice[:, 0]
                c_output[:, 1] = amp * x_slice[:, 1]

                coeff_layer = self.coeff_layers[c]
                ci = coeff_layer(c_output)
                c_real = ci[:, 0]
                c_imag = ci[:, 1]

                nonlin_output[:, id, c, 0] = c_real.squeeze(-1)  # (B,)
                nonlin_output[:, id, c, 1] = c_imag.squeeze(-1)

        return nonlin_output


class LinearInternal(nn.Module):
    def __init__(
            self, input_size, output_size, n_channels,
            batch_size, out_window=10
        ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_channels = n_channels
        self.out_window = out_window

        self.txa_filter_layers = TxaFilterEnsemble(
            n_channels, input_size, out_window
        )

        self.nlin_body = LinearInternalBody(
            out_window, output_size, n_channels
        )

        self.rxa_filter_layers = RxaFilterEnsemble(
            n_channels, out_window
        )

        self.bn_output = nn.BatchNorm1d(n_channels)  # For complex output

    def forward(self, x, h_0=None):
        filtered_x = self.txa_filter_layers(x)
        nonlin_output = self.nlin_body(filtered_x)
        output = self.rxa_filter_layers(nonlin_output)
        output = self.bn_output(output)

        return output

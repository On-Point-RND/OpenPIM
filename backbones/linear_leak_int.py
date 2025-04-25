import torch
import torch.nn as nn


class FiltLinear(nn.Module):
    def __init__(self, in_features, bias=False):
        super().__init__()
        self.filt_real = nn.Linear(in_features, 1, bias=bias)
        self.filt_imag = nn.Linear(in_features, 1, bias=bias)

    def forward(self, x_real, x_imag):
        return self.filt_real(x_real), self.filt_imag(x_imag)


class Linear(nn.Module):
    def __init__(self, input_size, output_size, n_channels, batch_size, out_window=10):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_channels = n_channels
        self.out_window = out_window

        # Batch normalization layers
        self.bn_output = nn.BatchNorm1d(n_channels)  # For complex output

        self.filter_layers_in = nn.ModuleList()
        for _ in range(n_channels):
            layer = FiltLinear(input_size - out_window + 1)
            self.filter_layers_in.append(layer)

        self.nonlin_layers = nn.ModuleList()
        for _ in range(n_channels):
            # layer = nn.Linear(output_size * (n_channels - 1), output_size, bias=False)
            layer = nn.Linear(output_size * n_channels, output_size, bias=False)
            self.nonlin_layers.append(layer)

        self.coeff = nn.Linear(output_size, output_size, bias=False)
        self.filter_layers_out = nn.ModuleList()
        for _ in range(n_channels):
            layer = FiltLinear(out_window)
            self.filter_layers_out.append(layer)

    def forward(self, x, h_0=None):
        # Input tensors of sizes B x N x C x 2
        B, N, C, _ = x.shape

        # INFO: resulting tensors of sizes B x C
        # Initialize output tensors
        x_init_real = x[:, :, :, 0]  # (B, N)
        x_init_imag = x[:, :, :, 1]  # (B, N)

        filtered_real = torch.zeros(
            (B, self.out_window, self.n_channels),
            device=x.device
        )
        filtered_imag = torch.zeros_like(filtered_real)

        for c, filt_layer in enumerate(self.filter_layers_in):
            x_real, x_imag = x_init_real[:, :, c], x_init_imag[:, :, c]
            for id in range(self.out_window):
                f_real, f_imag = filt_layer(
                    x_real[:, id:N-self.out_window+id+1],
                    x_imag[:, id:N-self.out_window+id+1]
                )

                amp = f_real.pow(2) + f_imag.pow(2)  # (B, N)
                f_real = amp * f_real
                f_imag = amp * f_imag

                filtered_real[:, id, c] = f_real.squeeze(-1)  # (B,)
                filtered_imag[:, id, c] = f_imag.squeeze(-1)

        nonlin_real = torch.zeros(
            (B, self.out_window, C),
            device=x.device
        )
        nonlin_imag = torch.zeros_like(nonlin_real)

        for id in range(self.out_window):
            for c, nonlin_layer in enumerate(self.nonlin_layers):
                filtered_real_c = torch.stack(
                    # [filtered_real[:, id, idx] for idx in range(C) if idx != c],
                    [filtered_real[:, id, idx] for idx in range(C)],
                    dim=-1
                )
                filtered_imag_c = torch.stack(
                    # [filtered_imag[:, id, idx] for idx in range(C) if idx != c],
                    [filtered_imag[:, id, idx] for idx in range(C)],
                    dim=-1
                )
                total_filtered = torch.cat([filtered_real_c, filtered_imag_c], dim=-1)

                c_output = nonlin_layer(total_filtered)

                c_real = c_output[:, 0]
                c_imag = c_output[:, 1]

                nonlin_real[:, id, c] = c_real.squeeze(-1)  # (B,)
                nonlin_imag[:, id, c] = c_imag.squeeze(-1)

        output = torch.zeros((B, self.n_channels, 2), device=x.device)
        for c, filt_layer in enumerate(self.filter_layers_out):
            out_real, out_imag = filt_layer(
                nonlin_real[:, :, c],
                nonlin_imag[:, :, c]
            )
            output[:, c, 0] = out_real.squeeze(-1)
            output[:, c, 1] = out_imag.squeeze(-1)

        output = self.bn_output(output)
        return output

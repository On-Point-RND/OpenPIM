import torch
import torch.nn as nn


class FiltLinear(nn.Module):
    def __init__(self, in_features, bias=False):
        super().__init__()
        self.filt_real = nn.Linear(in_features, 1, bias=bias)
        self.filt_imag = nn.Linear(in_features, 1, bias=bias)

    def forward(self, x_real, x_imag):
        return self.filt_real(x_real), self.filt_imag(x_imag)


class LinearExternal(nn.Module):
    def __init__(self, input_size, output_size, n_channels, batch_size, out_window=10):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_channels = n_channels
        self.out_window = out_window

        # Instance normalization at the beginning
        self.instance_norm_in_real = nn.InstanceNorm1d(n_channels)
        self.instance_norm_in_imag = nn.InstanceNorm1d(n_channels)

        # Batch normalization layer (existing)
        self.bn_output = nn.BatchNorm1d(n_channels)

        # Instance normalization at the end
        self.instance_norm_out = nn.InstanceNorm1d(2)  # For real and imaginary parts

        self.filter_layers_in = nn.ModuleList()
        for i in range(0, n_channels):
            layer = FiltLinear(input_size - out_window + 1)
            self.filter_layers_in.append(layer)

        self.nonlin_layers = nn.ModuleList()
        for i in range(0, n_channels):
            layer = nn.Linear(output_size * n_channels, output_size, bias=False)
            self.nonlin_layers.append(layer)

        self.coeff = nn.Linear(output_size, output_size, bias=False)
        self.filter_layers_out = nn.ModuleList()
        for i in range(0, n_channels):
            layer = FiltLinear(out_window)
            self.filter_layers_out.append(layer)

        self.amp1_weight = nn.Parameter(torch.tensor(0.001))
        self.amp2_weight = nn.Parameter(torch.tensor(0.001))
        self.amp3_weight = nn.Parameter(torch.tensor(0.001))

    def forward(self, x, h_0=None):
        B, N, C, _ = x.shape

        # INFO: resulting tensors of sizes B x C
        # Initialize output tensors
        x_init_real = x[:, :, :, 0]  # (B, N)
        x_init_imag = x[:, :, :, 1]  # (B, N)

        filtered_real = torch.zeros(
            (B, self.out_window, self.n_channels), device=x.device
        )
        filtered_imag = torch.zeros_like(filtered_real)

        for c, filt_layer in enumerate(self.filter_layers_in):
            x_real, x_imag = x_init_real[:, :, c], x_init_imag[:, :, c]
            for id in range(self.out_window):
                f_real, f_imag = filt_layer(
                    x_real[:, id : N - self.out_window + id + 1],
                    x_imag[:, id : N - self.out_window + id + 1],
                )

                filtered_real[:, id, c] = f_real.squeeze(-1)  # (B,)
                filtered_imag[:, id, c] = f_imag.squeeze(-1)

        nonlin_real = torch.zeros((B, self.out_window, C), device=x.device)
        nonlin_imag = torch.zeros_like(nonlin_real)

        for id in range(self.out_window):
            total_filtered = torch.cat(
                [filtered_real[:, id, :], filtered_imag[:, id, :]], dim=-1
            )

            for c, nonlin_layer in enumerate(self.nonlin_layers):
                c_output = nonlin_layer(total_filtered)

                c_real = c_output[:, 0]
                c_imag = c_output[:, 1]
                amp = c_real.pow(2) + c_imag.pow(2)  # (B, N)
                c_real = amp * c_real
                c_imag = amp * c_imag

                ci = self.coeff(torch.stack([c_real, c_imag], dim=-1))
                c_real = ci[:, 0]
                c_imag = ci[:, 1]

                nonlin_real[:, id, c] = c_real.squeeze(-1)  # (B,)
                nonlin_imag[:, id, c] = c_imag.squeeze(-1)

        output = torch.zeros((B, self.n_channels, 2), device=x.device)
        for c, filt_layer in enumerate(self.filter_layers_out):
            out_real, out_imag = filt_layer(nonlin_real[:, :, c], nonlin_imag[:, :, c])
            output[:, c, 0] = out_real.squeeze(-1)
            output[:, c, 1] = out_imag.squeeze(-1)
        output = self.bn_output(output)

        # Apply instance normalization at the end
        # output = output.permute(0, 2, 1)  # Shape: B x 2 x C
        # output = self.instance_norm_out(output)
        # output = output.permute(0, 2, 1)  # Shape: B x C x 2

        return output

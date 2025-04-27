import torch
import torch.nn as nn


class FiltLinear(nn.Module):
    def __init__(self, in_features, bias=False):
        super().__init__()
        self.weight_real = nn.Parameter(torch.randn(1, in_features))
        self.weight_imag = nn.Parameter(torch.randn(1, in_features))
        if bias:
            self.bias_real = nn.Parameter(torch.randn(1))
            self.bias_imag = nn.Parameter(torch.randn(1))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

    def forward(self, x_real, x_imag):
        real_out = torch.matmul(x_real, self.weight_real.t())
        imag_out = torch.matmul(x_imag, self.weight_imag.t())
        if self.bias_real is not None:
            real_out += self.bias_real
            imag_out += self.bias_imag
        return real_out, imag_out


class Linear(nn.Module):
    def __init__(self, input_size, output_size, n_channels, batch_size, out_window=10):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_channels = n_channels
        self.out_window = out_window

        self.bn_output = nn.BatchNorm1d(n_channels)

        # Create all filter layers with proper input features
        self.filter_layers_in = nn.ModuleList(
            [FiltLinear(input_size - out_window + 1) for _ in range(n_channels)]
        )

        # Create non-linear layers with correct input size
        self.nonlin_layers = nn.ModuleList(
            [
                nn.Linear(output_size * (n_channels - 1) * 2, output_size, bias=False)
                for _ in range(n_channels)
            ]
        )

        self.coeff = nn.Linear(output_size, output_size, bias=False)

        self.filter_layers_out = nn.ModuleList(
            [FiltLinear(out_window) for _ in range(n_channels)]
        )

    def forward(self, x, h_0=None):
        B, N, C, _ = x.shape

        # Initial split into real/imag
        x_real = x[..., 0]  # (B, N, C)
        x_imag = x[..., 1]  # (B, N, C)

        # Process input filters
        window_size = self.input_size - self.out_window + 1
        x_real_unfold = x_real.unfold(1, window_size, 1).permute(0, 2, 1, 3)
        x_imag_unfold = x_imag.unfold(1, window_size, 1).permute(0, 2, 1, 3)

        # Stack all filter weights
        in_weights_real = torch.cat(
            [layer.weight_real for layer in self.filter_layers_in]
        )
        in_weights_imag = torch.cat(
            [layer.weight_imag for layer in self.filter_layers_in]
        )

        # Compute all input filters at once
        filtered_real = torch.bmm(
            x_real_unfold.reshape(-1, window_size), in_weights_real.unsqueeze(-1)
        ).view(B, self.out_window, C)
        filtered_imag = torch.bmm(
            x_imag_unfold.reshape(-1, window_size), in_weights_imag.unsqueeze(-1)
        ).view(B, self.out_window, C)

        # Apply amplitude scaling
        amp = filtered_real.pow(2) + filtered_imag.pow(2)
        filtered_real = amp * filtered_real
        filtered_imag = amp * filtered_imag

        # Process non-linear layers
        # Create mask for excluding current channel
        mask = ~torch.eye(C, dtype=torch.bool, device=x.device)

        # Prepare input for non-linear layers
        nonlin_input = []
        for c in range(C):
            others_real = filtered_real[:, :, mask[c]]
            others_imag = filtered_imag[:, :, mask[c]]
            nonlin_input.append(torch.cat([others_real, others_imag], dim=-1))
        nonlin_input = torch.stack(nonlin_input, dim=2)

        # Apply all non-linear layers
        nonlin_weights = torch.cat(
            [layer.weight for layer in self.nonlin_layers], dim=0
        )
        nonlin_out = torch.matmul(
            nonlin_input.view(-1, nonlin_input.size(-1)), nonlin_weights.t()
        ).view(B, self.out_window, C, -1)

        # Apply coefficient layer
        coeff_out = self.coeff(nonlin_out)
        nonlin_real = coeff_out[..., 0]
        nonlin_imag = coeff_out[..., 1]

        # Process output filters
        out_weights_real = torch.cat(
            [layer.weight_real for layer in self.filter_layers_out]
        )
        out_weights_imag = torch.cat(
            [layer.weight_imag for layer in self.filter_layers_out]
        )

        # Reshape for batch matrix multiplication
        nonlin_real_flat = nonlin_real.permute(0, 2, 1).reshape(-1, self.out_window)
        nonlin_imag_flat = nonlin_imag.permute(0, 2, 1).reshape(-1, self.out_window)

        out_real_flat = torch.matmul(nonlin_real_flat, out_weights_real.t())
        out_imag_flat = torch.matmul(nonlin_imag_flat, out_weights_imag.t())

        # Reshape back to output format
        output_real = out_real_flat.view(B, C)
        output_imag = out_imag_flat.view(B, C)

        output = torch.stack([output_real, output_imag], dim=-1)
        output = self.bn_output(output.permute(0, 2, 1)).permute(0, 2, 1)

        return output

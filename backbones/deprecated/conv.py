import torch
import torch.nn as nn


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.fc_real = nn.Linear(in_features, out_features, bias=bias)
        self.fc_imag = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x_real, x_imag):
        real = self.fc_real(x_real) - self.fc_imag(x_imag)
        imag = self.fc_real(x_imag) + self.fc_imag(x_real)
        return real, imag


class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        # Real and imaginary convolution layers
        self.real_conv = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
        self.imag_conv = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)

        nn.init.dirac_(self.real_conv.weight)
        nn.init.dirac_(self.imag_conv.weight)

    def forward(self, x_real, x_imag):
        # Split into real and imaginary components
        real_part = x_real
        imag_part = x_imag

        # Apply convolutions separately
        real_out = self.real_conv(real_part)
        imag_out = self.imag_conv(imag_part)

        # Combine into complex output
        return real_out, imag_out


class SineActivation(nn.Module):
    def __init__(
        self, num_channels=1, scale_init=0.001, period_init=1.0, phase_init=0.0
    ):
        super().__init__()
        self.num_channels = num_channels

        # Initialize learnable parameters with proper broadcasting shapes
        self.scale = nn.Parameter(torch.tensor(scale_init))
        self.period = nn.Parameter(torch.tensor(period_init))
        self.phase = nn.Parameter(torch.tensor(phase_init))

    def forward(self, x):
        # x shape: (batch_size, num_channels, sequence_length)
        return self.scale * torch.sin(self.period * x + self.phase)


class PolynomialExpansion(nn.Module):
    def __init__(self, max_degree):
        super().__init__()
        self.max_degree = max_degree

    def forward(self, x_real, x_imag):
        x = torch.complex(x_real, x_imag)
        outputs_real = []
        outputs_imag = []
        for d in range(1, self.max_degree + 1):
            x_pow = x**d
            outputs_real.append(x_pow.real)
            outputs_imag.append(x_pow.imag)
        return torch.cat(outputs_real, dim=-1), torch.cat(outputs_imag, dim=-1)


class ConvModel(nn.Module):
    def __init__(self, input_size, output_size, n_channels, batch_size, poly_degree=4):
        super().__init__()
        self.n_channels = n_channels

        # Complex convolution layers with different kernel sizes
        self.conv3 = ComplexConv1d(n_channels, n_channels, kernel_size=3, padding=1)
        self.conv5 = ComplexConv1d(n_channels, n_channels, kernel_size=5, padding=2)
        self.conv8 = ComplexConv1d(n_channels, n_channels, kernel_size=9, padding=4)

        self.act_one_real = SineActivation()
        self.act_one_imag = SineActivation()

        self.conv_final = ComplexConv1d(
            n_channels * 3, n_channels, kernel_size=3, padding=1
        )

        self.act_two_real = SineActivation()
        self.act_two_imag = SineActivation()

        # self.poly_expand = PolynomialExpansion(poly_degree)
        # expanded_size = input_size * poly_degree * n_channels  # Compute expanded size
        # self.complex_fc = ComplexLinear(expanded_size, n_channels * output_size // 2)
        self.cross_channel_fc = ComplexLinear(n_channels * input_size, n_channels)

        self.bn_output = nn.BatchNorm1d(n_channels * 2)

        self.alpha = nn.Parameter(torch.tensor(0.01))

    def forward(self, x, h0):
        B, L, C, _ = x.shape  # Input shape: (B, C, L) complex

        x_r_init, x_i_init = x[..., 0].reshape(B, -1), x[..., 1].reshape(B, -1)
        x_real, x_imag = x[..., 0], x[..., 1]

        x_real, x_imag = x_real.reshape(B, C, L), x_imag.reshape(B, C, L)

        # Apply complex convolutions
        out_1 = self.conv3(x_real, x_imag)  # (B, n_channels, L) complex
        out_2 = self.conv5(x_real, x_imag)  # (B, n_channels, L) complex
        out_3 = self.conv8(x_real, x_imag)  # (B, n_channels, L) complex

        # Split and stack real/imag components from all kernels
        components_real = []
        components_imag = []
        for out in [out_1, out_2, out_3]:
            components_real.append(out[0])  # Real part
            components_imag.append(out[1])  # Imaginary part

        x_real = self.act_one_real(x_real)
        x_imag = self.act_one_imag(x_imag)

        # Concatenate along channel dimension: 6*n_channels total
        x_real = torch.cat(components_real, dim=1)  # (B, 6*n_channels, L)
        x_imag = torch.cat(components_imag, dim=1)  # (B, 6*n_channels, L)

        x_real = self.act_two_real(x_real)
        x_imag = self.act_two_imag(x_imag)

        x_real, x_imag = self.conv_final(x_real, x_real)

        x_real, x_imag = x_r_init + self.alpha * x_real.reshape(
            B, -1
        ), x_i_init + self.alpha * x_imag.reshape(B, -1)

        # x_real, x_imag = self.poly_expand(
        #     x_real, x_imag
        # )  # Output shape: (B, expanded_size)

        # # Further processing
        # x_real, x_imag = self.complex_fc(x_real, x_imag)

        # Cross-channel processing
        x_real = (
            x_real.view(B, self.n_channels, -1)
            .permute(0, 2, 1)
            .contiguous()
            .view(B, -1)
        )
        x_imag = (
            x_imag.view(B, self.n_channels, -1)
            .permute(0, 2, 1)
            .contiguous()
            .view(B, -1)
        )
        x_real, x_imag = self.cross_channel_fc(x_real, x_imag)

        output = torch.stack(
            [x_real.squeeze(-1), x_imag.squeeze(-1)], dim=-1
        )  # (B, C, 2)
        output = output.view(B, -1)  # (B, 2C)
        output = self.bn_output(output)  # Normalize
        output = output.view(B, C, 2)  # Back to (B, C, 2)
        # INFO: resulting tensors of sizes B x C x 2
        return output

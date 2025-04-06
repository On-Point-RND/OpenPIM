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


class ComplexActivation(nn.Module):
    def __init__(self, activation_type="split"):
        super().__init__()
        self.activation_type = activation_type

    def forward(self, real, imag):
        if self.activation_type == "split":
            # Split-complex activation
            return torch.tanh(real), torch.tanh(imag)
        elif self.activation_type == "modReLU":
            # ModReLU activation
            mod = torch.sqrt(real**2 + imag**2)
            phase = torch.atan2(imag, real)
            mod = torch.relu(mod)
            return mod * torch.cos(phase), mod * torch.sin(phase)


class ComplexPolynomial(nn.Module):
    def __init__(self, degree=3):
        super().__init__()
        self.degree = degree
        self.weights_real = nn.Parameter(torch.randn(degree))
        self.weights_imag = nn.Parameter(torch.randn(degree))

    def forward(self, real, imag):
        out_real = torch.zeros_like(real)
        out_imag = torch.zeros_like(imag)

        for d in range(self.degree):
            # Binomial expansion for complex numbers
            coef = self.weights_real[d] + 1j * self.weights_imag[d]
            terms_real = torch.zeros_like(real)
            terms_imag = torch.zeros_like(imag)

            for k in range(d + 1):
                real_term = torch.pow(real, d - k) * torch.pow(imag, k)
                if k % 2 == 0:
                    terms_real += real_term
                else:
                    terms_imag += real_term

            out_real += coef.real * terms_real - coef.imag * terms_imag
            out_imag += coef.real * terms_imag + coef.imag * terms_real

        return out_real, out_imag


class ComplexChannelAttention(nn.Module):
    def __init__(self, n_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_one = ComplexLinear(n_channels, n_channels // reduction_ratio)
        self.act = ComplexActivation()
        self.fc_last = ComplexLinear(n_channels // reduction_ratio, n_channels)

    def forward(self, x_real, x_imag):
        # x_real, x_imag shape: (B, N, C) where B is batch size, N is sequence length, C is channels
        b, n, c = x_real.shape

        # Transpose to (B, C, N) for 1D pooling
        x_real_t = x_real.transpose(1, 2)  # Shape: (B, C, N)
        x_imag_t = x_imag.transpose(1, 2)  # Shape: (B, C, N)

        # Apply pooling - output shape will be (B, C, 1)
        avg_real = self.avg_pool(x_real_t)
        avg_imag = self.avg_pool(x_imag_t)

        # Squeeze the last dimension and prepare for FC layer
        avg_real = avg_real.squeeze(-1)  # Shape: (B, C)
        avg_imag = avg_imag.squeeze(-1)  # Shape: (B, C)

        # Apply FC layers
        att_real, att_imag = self.fc_one(avg_real, avg_imag)
        att_real, att_imag = self.act(att_real, att_imag)
        att_real, att_imag = self.fc_last(att_real, att_imag)

        # Calculate attention weights
        att_weights = torch.sigmoid(torch.sqrt(att_real**2 + att_imag**2))

        # Apply attention - expand dimensions to match input
        out_real = x_real * att_weights.unsqueeze(1)  # Broadcasting to (B, N, C)
        out_imag = x_imag * att_weights.unsqueeze(1)  # Broadcasting to (B, N, C)

        return out_real, out_imag


class EnhancedLinear(nn.Module):
    def __init__(self, input_size, output_size, n_channels, batch_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_channels = n_channels

        # Batch normalization layers
        self.bn_input = nn.BatchNorm1d(n_channels * 2)
        self.bn_output = nn.BatchNorm1d(n_channels * 2)

        # Complex linear layers
        self.complex_fc_in = ComplexLinear(input_size, input_size)
        self.complex_fc = ComplexLinear(input_size, output_size // 2)

        # New components
        self.polynomial = ComplexPolynomial(degree=3)
        self.activation = ComplexActivation("modReLU")
        self.channel_attention = ComplexChannelAttention(n_channels)

    def forward(self, x, h_0=None):
        B, N, C, _ = x.shape

        # Extract real/imag components
        x_real = x[..., 0]  # (B, N, C)
        x_imag = x[..., 1]  # (B, N, C)

        # Apply channel attention
        x_real, x_imag = self.channel_attention(x_real, x_imag)

        # Initialize output tensors
        final_real = torch.zeros((B, self.n_channels), device=x.device)
        final_imag = torch.zeros_like(final_real)

        for c in range(self.n_channels):
            # Process each channel
            chan_real = x_real[:, :, c]  # (B, N)
            chan_imag = x_imag[:, :, c]  # (B, N)

            # Complex processing with non-linearities
            chan_real, chan_imag = self.complex_fc_in(chan_real, chan_imag)
            chan_real, chan_imag = self.activation(chan_real, chan_imag)
            chan_real, chan_imag = self.polynomial(chan_real, chan_imag)

            # Amplitude modulation
            amp = x[:, :, c, 0].pow(2) + x[:, :, c, 1].pow(2)
            chan_real = amp * chan_real
            chan_imag = amp * chan_imag

            # Final processing
            chan_real, chan_imag = self.complex_fc(chan_real, chan_imag)

            # Store results
            final_real[:, c] = chan_real.squeeze(-1)
            final_imag[:, c] = chan_imag.squeeze(-1)

        # Final formatting and normalization
        output = torch.stack([final_real, final_imag], dim=-1)
        output = output.view(B, -1)
        output = self.bn_output(output)
        output = output.view(B, C, 2)

        return output

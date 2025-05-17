import torch
import torch.nn as nn


import torch
import torch.nn as nn


class FiltLinear(nn.Module):
    def __init__(self, in_features, bias=False):
        super().__init__()
        self.filt_real = nn.Linear(in_features, 1, bias=bias)
        self.filt_imag = nn.Linear(in_features, 1, bias=bias)

    def forward(self, x_real, x_imag):
        return self.filt_real(x_real), self.filt_imag(x_imag)


class ComplexScaling(nn.Module):
    def __init__(self, n_channels):
        """
        Initialize complex scaling layer.
        Args:
            n_channels (int): Number of channels to scale independently
            with their own complex scaling factor
        """
        super().__init__()
        self.n_channels = n_channels

        # Create learnable weights for each channel,
        # Each channel has its own complex scaling factor
        # (real and imag parts)
        self.weights = nn.Parameter(torch.randn(n_channels, 2))

    def forward(self, x):
        """
        Apply complex scaling to input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape:
                            - (batch_size, seq_len, n_channels, 2) or
                            - (batch_size, n_channels, 2)
                            where last dimension contains (real, imag) components

        Returns:
            torch.Tensor: Scaled complex numbers with the same shape as input
        """
        # Get real and imaginary parts
        real = x[..., 0]  # shape: (batch_size, [seq_len,] n_channels)
        imag = x[..., 1]  # shape: (batch_size, [seq_len,] n_channels)

        # Get the number of dimensions to determine input shape
        ndim = x.dim()

        # Prepare weights for broadcasting
        if ndim == 3:  # (batch_size, n_channels, 2)
            w_real = self.weights[:, 0].unsqueeze(0)  # shape: (1, n_channels)
            w_imag = self.weights[:, 1].unsqueeze(0)  # shape: (1, n_channels)
        else:  # (batch_size, seq_len, n_channels, 2)
            w_real = (
                self.weights[:, 0].unsqueeze(0).unsqueeze(0)
            )  # shape: (1, 1, n_channels)
            w_imag = (
                self.weights[:, 1].unsqueeze(0).unsqueeze(0)
            )  # shape: (1, 1, n_channels)

        # Apply complex multiplication for each channel
        # For each channel c:
        # new_real = w_real[c] * real - w_imag[c] * imag
        # new_imag = w_real[c] * imag + w_imag[c] * real
        new_real = w_real * real - w_imag * imag
        new_imag = w_real * imag + w_imag * real

        # Reshape
        if ndim == 3:
            new_real = new_real.unsqueeze(-1)  # shape: (batch_size, n_channels, 1)
            new_imag = new_imag.unsqueeze(-1)  # shape: (batch_size, n_channels, 1)
        else:
            new_real = new_real.unsqueeze(
                -1
            )  # shape: (batch_size, seq_len, n_channels, 1)
            new_imag = new_imag.unsqueeze(
                -1
            )  # shape: (batch_size, seq_len, n_channels, 1)

        return torch.cat([new_real, new_imag], dim=-1)


class TxaFilterEnsemble(nn.Module):
    def __init__(self, n_channels, in_seq_size, out_seq_size):
        super().__init__()
        self.n_channels = n_channels
        self.out_seq_size = out_seq_size
        self.txa_filter_layers = nn.ModuleList()
        for _ in range(n_channels):
            layer = FiltLinear(in_seq_size - out_seq_size + 1)
            self.txa_filter_layers.append(layer)

    def forward(self, x):
        n_batch, n_timesteps, _, _ = x.shape
        output = torch.empty(
            (n_batch, self.out_seq_size, self.n_channels, 2), device=x.device
        )
        for c, filt_layer in enumerate(self.txa_filter_layers):
            for id in range(self.out_seq_size):
                f_real, f_imag = filt_layer(
                    x[:, id : n_timesteps - self.out_seq_size + id + 1, c, 0],
                    x[:, id : n_timesteps - self.out_seq_size + id + 1, c, 1],
                )
                output[:, id, c, 0] = f_real.squeeze(-1)
                output[:, id, c, 1] = f_imag.squeeze(-1)
        return output


class TxaFilterEnsembleTorch(nn.Module):
    def __init__(self, n_channels, in_seq_size, out_seq_size):
        super().__init__()
        self.n_channels = n_channels
        self.out_seq_size = out_seq_size
        self.conv_size = in_seq_size - out_seq_size + 1
        self.txa_filter_layers = nn.ModuleList()
        for _ in range(n_channels):
            layer = nn.Conv1d(
                in_channels=2,
                out_channels=2,
                kernel_size=self.conv_size,
                padding="valid",
                groups=1,
                bias=False,
            )
            self.txa_filter_layers.append(layer)

    def forward(self, x):
        n_batch, n_seq, *_ = x.shape
        out_seq_len = n_seq - self.conv_size + 1
        output = torch.empty(
            (n_batch, out_seq_len, self.n_channels, 2), device=x.device
        )
        for c, conv_layer in enumerate(self.txa_filter_layers):
            channel_data = x[:, :, c, :]
            channel_data = channel_data.transpose(2, 1)
            y = conv_layer(channel_data)
            y = y.transpose(2, 1)
            output[:, :, c, :] = y
        return output


class RxaFilterEnsemble(nn.Module):
    def __init__(self, n_channels, seq_size):
        super().__init__()
        self.n_channels = n_channels
        self.seq_size = seq_size
        self.rxa_filter_layers = nn.ModuleList()
        for _ in range(n_channels):
            layer = FiltLinear(seq_size)
            self.rxa_filter_layers.append(layer)

    def forward(self, x):
        n_batch, _, _, _ = x.shape
        output = torch.empty((n_batch, self.n_channels, 2), device=x.device)
        for c, filt_layer in enumerate(self.rxa_filter_layers):
            out_real, out_imag = filt_layer(x[:, :, c, 0], x[:, :, c, 1])
            output[:, c, 0] = out_real.squeeze(-1)
            output[:, c, 1] = out_imag.squeeze(-1)
        return output


class RxaFilterEnsembleTorch(nn.Module):
    def __init__(self, n_channels, seq_size):
        super().__init__()
        self.n_channels = n_channels
        self.conv_size = seq_size
        self.rxa_filter_layers = nn.ModuleList()
        for _ in range(n_channels):
            layer = nn.Conv1d(
                in_channels=2,
                out_channels=2,
                kernel_size=seq_size,
                padding="valid",
                groups=1,
                bias=False,
            )
            self.rxa_filter_layers.append(layer)

    def forward(self, x):
        n_batch, n_seq, *_ = x.shape
        out_seq_len = n_seq - self.conv_size + 1
        output = torch.empty(
            (n_batch, out_seq_len, self.n_channels, 2), device=x.device
        )
        for c, conv_layer in enumerate(self.rxa_filter_layers):
            channel_data = x[:, :, c, :]
            channel_data = channel_data.transpose(2, 1)
            y = conv_layer(channel_data)
            y = y.transpose(2, 1)
            output[:, :, c, :] = y
        return output[:, 0, :, :]


class MediumSimulation(nn.Module):
    def __init__(self, n_channels, conv_size):
        super().__init__()
        self.n_channels = n_channels
        self.conv_size = conv_size
        self.conv_layers = nn.ModuleList()
        for _ in range(n_channels):
            layer = nn.Conv1d(
                in_channels=2,
                out_channels=2,
                kernel_size=conv_size,
                padding="valid",
                groups=1,
                bias=False,
            )
            self.conv_layers.append(layer)

    def forward(self, x):
        n_batch, n_seq, *_ = x.shape
        out_seq_len = n_seq - self.conv_size + 1
        output = torch.empty(
            (n_batch, out_seq_len, self.n_channels, 2), device=x.device
        )
        for c, conv_layer in enumerate(self.conv_layers):
            channel_data = x[:, :, c, :]
            channel_data = channel_data.transpose(2, 1)
            y = conv_layer(channel_data)
            y = y.transpose(2, 1)
            output[:, :, c, :] = y
        return output


################## COMPLEX


class TxaFilterComplexTorch(nn.Module):
    def __init__(self, n_channels, in_seq_size, out_seq_size):
        super().__init__()
        self.n_channels = n_channels
        self.out_seq_size = out_seq_size
        self.conv_size = in_seq_size - out_seq_size + 1
        self.txa_filter_layers = nn.ModuleList()
        for _ in range(n_channels):
            layer = nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=self.conv_size,
                padding="valid",
                groups=1,
                bias=False,
                dtype=torch.complex64,
            )
            self.txa_filter_layers.append(layer)

    def forward(self, x):
        x = x[..., 0] + 1j * x[..., 1]
        x = x.unsqueeze(-1)
        n_batch, n_seq, *_ = x.shape
        out_seq_len = n_seq - self.conv_size + 1
        output = torch.empty(
            (n_batch, out_seq_len, self.n_channels, 1),
            device=x.device,
            dtype=torch.complex64,
        )
        for c, conv_layer in enumerate(self.txa_filter_layers):
            channel_data = x[:, :, c, :]
            channel_data = channel_data.transpose(2, 1)
            y = conv_layer(channel_data)
            y = y.transpose(2, 1)
            output[:, :, c, :] = y
        output = torch.stack((output.real, output.imag), dim=-1)
        return output


class RxaFilterComplexTorch(nn.Module):
    def __init__(self, n_channels, seq_size):
        super().__init__()
        self.n_channels = n_channels
        self.conv_size = seq_size
        self.rxa_filter_layers = nn.ModuleList()
        for _ in range(n_channels):
            layer = nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=seq_size,
                padding="valid",
                groups=1,
                bias=False,
                dtype=torch.complex64,
            )
            self.rxa_filter_layers.append(layer)

    def forward(self, x):
        x = x[..., 0] + 1j * x[..., 1]
        n_batch, n_seq, *_ = x.shape
        out_seq_len = n_seq - self.conv_size + 1
        output = torch.empty(
            (n_batch, out_seq_len, self.n_channels, 1),
            device=x.device,
            dtype=torch.complex64,
        )
        for c, conv_layer in enumerate(self.rxa_filter_layers):
            channel_data = x[:, :, c, :]

            channel_data = channel_data.transpose(2, 1)

            y = conv_layer(channel_data)
            y = y.transpose(2, 1)
            output[:, :, c] = y
        output = torch.stack((output.real, output.imag), dim=-1)
        return output[:, 0, :, :]


class MediumSimulationComplex(nn.Module):
    def __init__(self, n_channels, conv_size):
        super().__init__()
        self.n_channels = n_channels
        self.conv_size = conv_size
        self.conv_layers = nn.ModuleList()
        for _ in range(n_channels):
            layer = nn.Conv1d(
                in_channels=2,
                out_channels=2,
                kernel_size=conv_size,
                padding="valid",
                groups=1,
                bias=False,
            )
            self.conv_layers.append(layer)

    def forward(self, x):
        n_batch, n_seq, *_ = x.shape
        out_seq_len = n_seq - self.conv_size + 1
        output = torch.empty(
            (n_batch, out_seq_len, self.n_channels, 1),
            device=x.device,
            dtype=torch.complex64,
        )
        for c, conv_layer in enumerate(self.conv_layers):
            channel_data = x[:, :, c, :]
            channel_data = channel_data.transpose(2, 1)
            y = conv_layer(channel_data)
            y = y.transpose(2, 1)
            output[:, :, c, :] = y
        return output

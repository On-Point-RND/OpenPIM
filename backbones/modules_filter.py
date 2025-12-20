import torch
import torch.nn as nn


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


### COMPLEX

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

import torch
import torch.nn as nn


class FiltLinear(nn.Module):
    def __init__(self, in_features, bias=False):
        super().__init__()
        self.filt_real = nn.Linear(in_features, 1, bias=bias)
        self.filt_imag = nn.Linear(in_features, 1, bias=bias)

    def forward(self, x_real, x_imag):
        return self.filt_real(x_real), self.filt_imag(x_imag)


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

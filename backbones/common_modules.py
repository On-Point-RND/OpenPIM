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
            n_channels (int): Number of channels to process independently
        """
        super().__init__()
        self.n_channels = n_channels

        # Create learnable weights for each channel
        # Each channel has its own complex scaling factor (real and imag parts)
        self.weights = nn.Parameter(torch.randn(n_channels, 2))

    def forward(self, x):
        """
        Apply complex scaling to input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_channels, 2)
                            where last dimension contains (real, imag) components

        Returns:
            torch.Tensor: Scaled complex numbers with same shape as input
        """
        # Get real and imaginary parts
        real = x[..., 0]  # shape: (batch_size, seq_len, n_channels)
        imag = x[..., 1]  # shape: (batch_size, seq_len, n_channels)

        # Apply complex multiplication for each channel
        # For each channel c:
        # new_real = w_real[c] * real - w_imag[c] * imag
        # new_imag = w_real[c] * imag + w_imag[c] * real
        new_real = self.weights[:, 0].unsqueeze(0).unsqueeze(0) * real - \
                   self.weights[:, 1].unsqueeze(0).unsqueeze(0) * imag
        new_imag = self.weights[:, 0].unsqueeze(0).unsqueeze(0) * imag + \
                   self.weights[:, 1].unsqueeze(0).unsqueeze(0) * real

        # Reshape and concatenate instead of stacking
        b, s, c = new_real.shape
        new_real = new_real.reshape(b, s, c, 1)
        new_imag = new_imag.reshape(b, s, c, 1)
        return torch.cat([new_real, new_imag], dim=-1)


class TxaFilterEnsemble(nn.Module):
    def __init__(self, n_channels, input_size, out_window):
        super().__init__()
        self.n_channels = n_channels
        self.out_window = out_window
        self.txa_filter_layers = nn.ModuleList()
        for _ in range(n_channels):
            layer = FiltLinear(input_size - out_window + 1)
            self.txa_filter_layers.append(layer)
            
    def forward(self, x):
        n_batch, n_timesteps, _, _ = x.shape
        output = torch.empty(
            (n_batch, self.out_window, self.n_channels, 2),
            device=x.device
        )
        for c, filt_layer in enumerate(self.txa_filter_layers):
            for id in range(self.out_window):
                f_real, f_imag = filt_layer(
                    x[:, id : n_timesteps - self.out_window + id + 1, c, 0],
                    x[:, id : n_timesteps - self.out_window + id + 1, c, 1]
                )
                output[:, id, c, 0] = f_real.squeeze(-1)
                output[:, id, c, 1] = f_imag.squeeze(-1)
        return output


class RxaFilterEnsemble(nn.Module):
    def __init__(self, n_channels, timelag_size):
        super().__init__()
        self.n_channels = n_channels
        self.timelag_size = timelag_size
        
        self.rxa_filter_layers = nn.ModuleList()
        for _ in range(n_channels):
            layer = FiltLinear(timelag_size)
            self.rxa_filter_layers.append(layer)
            
    def forward(self, x):
        n_batch, _, _, _ = x.shape
        output = torch.zeros(
            (n_batch, self.n_channels, 2),
            device=x.device
        )
        for c, filt_layer in enumerate(self.rxa_filter_layers):
            out_real, out_imag = filt_layer(
                x[:, :, c, 0], x[:, :, c, 1]
            )
            output[:, c, 0] = out_real.squeeze(-1)
            output[:, c, 1] = out_imag.squeeze(-1)
        return output

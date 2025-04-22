import torch
from torch import nn


class FiltLinear(nn.Module):
    def __init__(self, in_features, bias=False):
        super().__init__()
        self.filt_real = nn.Linear(in_features, 1, bias=bias)
        self.filt_imag = nn.Linear(in_features, 1, bias=bias)

    def forward(self, x_real, x_imag):
        return self.filt_real(x_real), self.filt_imag(x_imag)


class Simple(nn.Module):
    def __init__(
        self,
        hidden_size,
        output_size,
        num_layers,
        batch_size,
        bidirectional=False,
        batch_first=True,
        bias=True,
        input_len=1,
        n_channels=1,
    ):
        super(Simple, self).__init__()
        self.n_channels = n_channels
        self.hidden_size = hidden_size
        # Instance normalization at the beginning

        # Existing layers (unchanged)
        self.filter_layers_in = nn.ModuleList(
            [FiltLinear(input_len) for _ in range(n_channels)]
        )

        self.amp_weight = nn.Parameter(torch.tensor(torch.ones(16)))
        self.amp2_weight = nn.Parameter(torch.tensor(torch.ones(16)))

        self.scale = nn.Parameter(torch.ones(1, self.n_channels, 1))  # Shape: (1, C, 2)

    def forward(self, x, h_0):
        B, L, C, _ = x.shape

        x_init_real = x[:, -1, :, 0]  # (B, L, C)
        x_init_imag = x[:, -1, :, 1]  # (B, L, C)

        # filtered_real = torch.zeros((B, self.n_channels), device=x.device)
        # filtered_imag = torch.zeros_like(filtered_real)
        # for c, filt_layer in enumerate(self.filter_layers_in):
        #     x_real, x_imag = x_init_real[:, :, c], x_init_imag[:, :, c]
        #     f_real, f_imag = filt_layer(x_real, x_imag)
        #     filtered_real[:, c] = f_real.squeeze(-1)
        #     filtered_imag[:, c] = f_imag.squeeze(-1)

        amp2 = x_init_real.pow(2) + x_init_imag.pow(2)

        amp_scaled = self.amp_weight * amp2 ** (1 / 2)

        amp2_scaled = self.amp_weight * amp2

        i = (amp2_scaled * x_init_real + amp_scaled * x_init_real).unsqueeze(-1)
        q = (amp2_scaled * x_init_imag + amp_scaled * x_init_imag).unsqueeze(-1)
        #        print(i.shape, q.shape)

        out = torch.cat([self.scale * i, self.scale * q], dim=-1)
        return out

import torch
from torch import nn


class FiltLinear(nn.Module):
    def __init__(self, in_features, bias=False):
        super().__init__()
        self.filt_real = nn.Linear(in_features, 1, bias=bias)
        self.filt_imag = nn.Linear(in_features, 1, bias=bias)

    def forward(self, x_real, x_imag):
        return self.filt_real(x_real), self.filt_imag(x_imag)


class DGRU(nn.Module):
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
        super(DGRU, self).__init__()
        # ... (existing parameters remain the same)
        self.n_channels = n_channels
        self.hidden_size = hidden_size
        # Instance normalization at the beginning
        self.instance_norm_in_real = nn.InstanceNorm1d(n_channels)
        self.instance_norm_in_imag = nn.InstanceNorm1d(n_channels)

        # # Instance normalization at the end
        self.instance_norm_out = nn.InstanceNorm1d(2)  # For real/imag parts

        # Existing layers (unchanged)
        self.filter_layers_in = nn.ModuleList(
            [FiltLinear(input_len) for _ in range(n_channels)]
        )
        self.rnn = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size * 4,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=batch_first,
            bias=bias,
        )
        self.fc_hid = nn.Linear(n_channels * 6, hidden_size * 2, bias=bias)
        self.fc_out = nn.Linear(hidden_size * 4, output_size * n_channels, bias=bias)
        self.amp1_weight = nn.Parameter(torch.tensor(0.001))
        self.amp2_weight = nn.Parameter(torch.tensor(0.001))
        self.amp3_weight = nn.Parameter(torch.tensor(0.001))

        self.act = nn.Tanh()

    def forward(self, x, h_0):
        B, L, C, _ = x.shape

        # Split and normalize input
        x_init_real = x[:, :, :, 0]  # (B, L, C)
        x_init_imag = x[:, :, :, 1]  # (B, L, C)

        # Apply instance normalization at the beginning
        x_init_real = self.instance_norm_in_real(x_init_real.permute(0, 2, 1)).permute(
            0, 2, 1
        )
        x_init_imag = self.instance_norm_in_imag(x_init_imag.permute(0, 2, 1)).permute(
            0, 2, 1
        )

        # Existing filtering and processing
        filtered_real = torch.zeros((B, self.n_channels), device=x.device)
        filtered_imag = torch.zeros_like(filtered_real)
        for c, filt_layer in enumerate(self.filter_layers_in):
            x_real, x_imag = x_init_real[:, :, c], x_init_imag[:, :, c]
            f_real, f_imag = filt_layer(x_real, x_imag)
            filtered_real[:, c] = self.act(f_real.squeeze(-1))
            filtered_imag[:, c] = self.act(f_imag.squeeze(-1))

        amp2 = filtered_real.pow(2) + filtered_imag.pow(2)
        amp = amp2.sqrt()
        amp3 = amp.pow(3)
        amp_scaled = self.amp1_weight * amp
        amp2_scaled = self.amp2_weight * amp2
        #  amp3_scaled = self.amp3_weight * amp3

        x = torch.cat(
            (
                amp2_scaled * filtered_real,
                amp2_scaled * filtered_imag,
                amp_scaled * filtered_real,
                amp_scaled * filtered_imag,
                filtered_real,
                filtered_imag,
            ),
            dim=-1,
        )

        # Regression and output
        out = self.fc_hid(x)
        out, _ = self.rnn(
            out.unsqueeze(0), torch.zeros(1, 1, self.hidden_size * 4).to(x.device)
        )
        out = out.view(B, -1)
        out = self.fc_out(out)
        out = out.view(B, C, 2)  # (B, C, 2)

        #  Apply instance normalization at the end
        out = out.permute(0, 2, 1)  # (B, 2, C)
        out = self.instance_norm_out(out)
        out = out.permute(0, 2, 1)  # (B, C, 2)

        return out

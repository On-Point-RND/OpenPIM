import torch
from torch import nn


class FiltLinear(nn.Module):
    def __init__(self, in_features, bias=False):
        super().__init__()
        self.filt_real = nn.Linear(in_features, 1, bias=bias)
        self.filt_imag = nn.Linear(in_features, 1, bias=bias)

    def forward(self, x_real, x_imag):
        return self.filt_real(x_real), self.filt_imag(x_imag)


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features=1, bias=False):
        super().__init__()
        self.fc_real = nn.Linear(in_features, out_features, bias=bias)
        self.fc_imag = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x_real, x_imag):
        real = self.fc_real(x_real) - self.fc_imag(x_imag)
        imag = self.fc_real(x_imag) + self.fc_imag(x_real)
        return real, imag


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

        self.amp_weight = nn.Parameter(torch.tensor(1.0))
        self.amp2_weight = nn.Parameter(torch.tensor(1.0))

        self.scale = nn.Parameter(torch.ones(1, self.n_channels))  # Shape: (1, C, 2)

    def forward(self, x, h_0):
        B, L, C, _ = x.shape

        x_init_real = x[:, :, :, 0]  # (B, L, C)
        x_init_imag = x[:, :, :, 1]  # (B, L, C)

        amp2 = x_init_real.pow(2) + x_init_imag.pow(2)

        amp_scaled = self.amp_weight * amp2 ** (1 / 2)

        amp2_scaled = self.amp_weight * amp2

        x_init_real = amp2_scaled * x_init_real + amp_scaled * x_init_real
        x_init_imag = amp2_scaled * x_init_imag + amp_scaled * x_init_imag
        #        print(i.shape, q.shape)

        # print(x_init_imag.shape, x_init_real.shape, amp2_scaled.shape)
        filtered_real = torch.zeros((B, self.n_channels), device=x.device)
        filtered_imag = torch.zeros_like(filtered_real)
        for c, filt_layer in enumerate(self.filter_layers_in):
            x_real, x_imag = x_init_real[:, :, c], x_init_imag[:, :, c]
            f_real, f_imag = filt_layer(x_real, x_imag)
            filtered_real[:, c] = f_real.squeeze(-1)
            filtered_imag[:, c] = f_imag.squeeze(-1)

        out = torch.cat(
            [
                (self.scale * filtered_real).unsqueeze(-1),
                (self.scale * filtered_imag).unsqueeze(-1),
            ],
            dim=-1,
        )

        return out


class SimpleConv(nn.Module):
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
        super(SimpleConv, self).__init__()
        self.n_channels = n_channels

        # New: Complex-valued 2D convolution for channel-time interactions
        self.conv_real = nn.Conv1d(
            in_channels=n_channels,  # Process real/imag separately [[2]][[5]]
            out_channels=n_channels,
            kernel_size=3,  # Channels × temporal window [[4]]
            padding=1,  # Maintain temporal dimension [[8]]
            padding_mode="replicate",
        )
        self.conv_imag = nn.Conv1d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
        )

        # Existing layers (unchanged)
        self.filter_layers_in = nn.ModuleList(
            [FiltLinear(input_len) for _ in range(n_channels)]
        )

        self.amp_weight = nn.Parameter(torch.tensor(1.0))
        self.amp2_weight = nn.Parameter(torch.tensor(1.0))

        self.scale = nn.Parameter(torch.ones(1, self.n_channels))  # Shape: (1, C, 2)

    def forward(self, x, h_0):
        B, L, C, _ = x.shape

        # Split complex components
        x_real = x[..., 0]  # (B, L, C)
        x_imag = x[..., 1]  # (B, L, C)

        x_real_conv = self.conv_real(x_real.reshape(B, C, L)).reshape(B, L, C)

        # .reshape(B, L, C)
        # (B, L, 1) → broadcast to C
        x_imag_conv = self.conv_imag(x_imag.reshape(B, C, L)).reshape(B, L, C)
        # Restore channel dimension through broadcasting [[7]]
        x_real = x_real + torch.relu(x_real_conv.expand(-1, -1, C))
        x_imag = x_imag + torch.relu(x_imag_conv.expand(-1, -1, C))

        # Existing amplitude scaling logic
        amp2 = x_real.pow(2) + x_imag.pow(2)
        amp_scaled = self.amp_weight * amp2 ** (1 / 2)
        amp2_scaled = self.amp_weight * amp2

        x_real = amp2_scaled * x_real + amp_scaled * x_real
        x_imag = amp2_scaled * x_imag + amp_scaled * x_imag

        # Existing per-channel filtering
        filtered_real = torch.zeros((B, self.n_channels), device=x.device)
        filtered_imag = torch.zeros_like(filtered_real)
        for c, filt_layer in enumerate(self.filter_layers_in):
            f_real, f_imag = filt_layer(x_real[:, :, c], x_imag[:, :, c])
            filtered_real[:, c] = f_real.squeeze(-1)
            filtered_imag[:, c] = f_imag.squeeze(-1)

        out = torch.cat(
            [
                (self.scale * filtered_real).unsqueeze(-1),
                (self.scale * filtered_imag).unsqueeze(-1),
            ],
            dim=-1,
        )

        return out


class SimpleMixing(nn.Module):
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
        super(SimpleMixing, self).__init__()
        self.n_channels = n_channels
        self.hidden_size = hidden_size

        # Add inter-channel mixing layers
        self.mixing_real = nn.Linear(n_channels, n_channels, bias=False)
        self.mixing_imag = nn.Linear(n_channels, n_channels, bias=False)

        # Existing layers (unchanged)
        self.filter_layers_in = nn.ModuleList(
            [FiltLinear(input_len) for _ in range(n_channels)]
        )

        self.amp_weight = nn.Parameter(torch.tensor(1.0))
        self.amp2_weight = nn.Parameter(torch.tensor(1.0))

        self.scale = nn.Parameter(torch.ones(1, self.n_channels))  # Shape: (1, C, 2)

    def forward(self, x, h_0):
        B, L, C, _ = x.shape

        x_init_real = x[:, :, :, 0]  # (B, L, C)
        x_init_imag = x[:, :, :, 1]  # (B, L, C)

        # Apply inter-channel mixing
        x_real_mixed = self.mixing_real(x_init_real)  # (B, L, C)
        x_imag_mixed = self.mixing_imag(x_init_imag)  # (B, L, C)

        # Compute nonlinear terms from mixed signals
        amp2 = x_real_mixed.pow(2) + x_imag_mixed.pow(2)
        amp = amp2**0.5

        amp_scaled = self.amp_weight * amp
        amp2_scaled = self.amp2_weight * amp2

        # Apply nonlinear scaling
        x_real = amp2_scaled * x_real_mixed + amp_scaled * x_real_mixed
        x_imag = amp2_scaled * x_imag_mixed + amp_scaled * x_imag_mixed

        # Process each channel through its filter
        filtered_real = torch.zeros((B, self.n_channels), device=x.device)
        filtered_imag = torch.zeros_like(filtered_real)
        for c, filt_layer in enumerate(self.filter_layers_in):
            # Extract c-th channel data
            x_real_c = x_real[:, :, c]  # (B, L)
            x_imag_c = x_imag[:, :, c]  # (B, L)
            f_real, f_imag = filt_layer(x_real_c, x_imag_c)
            filtered_real[:, c] = f_real.squeeze(-1)
            filtered_imag[:, c] = f_imag.squeeze(-1)

        out = torch.cat(
            [
                (self.scale * filtered_real).unsqueeze(-1),
                (self.scale * filtered_imag).unsqueeze(-1),
            ],
            dim=-1,
        )

        return out


class SimpleNM(nn.Module):
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
        super(SimpleNM, self).__init__()
        self.n_channels = n_channels
        self.hidden_size = hidden_size

        # New: Nonlinear cross-channel interaction layers
        self.channel_mixer_real = nn.Sequential(
            nn.Linear(n_channels, hidden_size),
            nn.ReLU(),  # Captures nonlinear interactions [[2]][[8]]
            nn.Linear(hidden_size, n_channels),
        )
        self.channel_mixer_imag = nn.Sequential(
            nn.Linear(n_channels, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_channels),
        )

        # Existing layers (unchanged)
        self.filter_layers_in = nn.ModuleList(
            [FiltLinear(input_len) for _ in range(n_channels)]
        )

        self.amp_weight = nn.Parameter(torch.tensor(1.0))
        self.amp2_weight = nn.Parameter(torch.tensor(1.0))

        self.scale = nn.Parameter(torch.ones(1, self.n_channels))

    def forward(self, x, h_0):
        B, L, C, _ = x.shape

        x_init_real = x[:, :, :, 0]  # (B, L, C)
        x_init_imag = x[:, :, :, 1]  # (B, L, C)

        # Apply nonlinear cross-channel mixing
        x_init_real = self.channel_mixer_real(x_init_real)  # (B, L, C) [[2]][[8]]
        x_init_imag = self.channel_mixer_imag(x_init_imag)  # (B, L, C)

        amp2 = x_init_real.pow(2) + x_init_imag.pow(2)  # Nonlinear term after mixing

        amp_scaled = self.amp_weight * amp2 ** (1 / 2)
        amp2_scaled = self.amp_weight * amp2

        x_init_real = amp2_scaled * x_init_real + amp_scaled * x_init_real
        x_init_imag = amp2_scaled * x_init_imag + amp_scaled * x_init_imag

        filtered_real = torch.zeros((B, self.n_channels), device=x.device)
        filtered_imag = torch.zeros_like(filtered_real)
        for c, filt_layer in enumerate(self.filter_layers_in):
            x_real, x_imag = x_init_real[:, :, c], x_init_imag[:, :, c]
            f_real, f_imag = filt_layer(x_real, x_imag)
            filtered_real[:, c] = f_real.squeeze(-1)
            filtered_imag[:, c] = f_imag.squeeze(-1)

        out = torch.cat(
            [
                (self.scale * filtered_real).unsqueeze(-1),
                (self.scale * filtered_imag).unsqueeze(-1),
            ],
            dim=-1,
        )

        return out


class SimpleC2(nn.Module):
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
        super(SimpleC2, self).__init__()
        self.n_channels = n_channels
        self.hidden_size = hidden_size

        # Channel interaction convolution (kernel_size = [time_window, channels])
        self.channel_mixer_real = nn.Conv2d(
            in_channels=1,
            out_channels=n_channels,
            kernel_size=(3, n_channels),  # 3 time steps, full channel width
            padding=(1, 0),  # Maintain temporal dimension length
        )
        self.channel_mixer_imag = nn.Conv2d(
            in_channels=1,
            out_channels=n_channels,
            kernel_size=(3, n_channels),
            padding=(1, 0),
        )

        # Existing processing layers
        self.filter_layers_in = nn.ModuleList(
            [FiltLinear(input_len) for _ in range(n_channels)]
        )

        # Enhanced parameterization
        self.amp_weight = nn.Parameter(torch.ones(n_channels))  # Per-channel scaling
        self.amp2_weight = nn.Parameter(torch.ones(n_channels))
        self.scale = nn.Parameter(torch.ones(1, n_channels, 2))  # Combined scaling

    def forward(self, x, h_0):
        B, L, C, _ = x.shape

        # Separate real/imaginary components
        x_real = x[..., 0]  # (B, L, C)
        x_imag = x[..., 1]

        # Channel mixing convolution
        def apply_mixer(x, mixer):
            x = x.unsqueeze(1)  # Add channel dim (B, 1, L, C)
            x = mixer(x)  # (B, C, L, 1)
            return x.squeeze(-1).permute(0, 2, 1)  # (B, L, C)

        mixed_real = apply_mixer(x_real, self.channel_mixer_real)
        mixed_imag = apply_mixer(x_imag, self.channel_mixer_imag)

        # Nonlinear processing with per-channel parameters
        amp = (mixed_real**2 + mixed_imag**2 + 1e-8).sqrt()
        amp_scaled = self.amp_weight * amp
        amp2_scaled = self.amp2_weight * amp**2

        # Nonlinear interaction
        processed_real = (amp_scaled + amp2_scaled) * mixed_real
        processed_imag = (amp_scaled + amp2_scaled) * mixed_imag

        # Channel-wise filtering
        filtered_real = torch.zeros((B, C), device=x.device)
        filtered_imag = torch.zeros_like(filtered_real)

        for c in range(C):
            r, i = self.filter_layers_in[c](
                processed_real[:, :, c], processed_imag[:, :, c]
            )
            filtered_real[:, c] = r.squeeze(-1)
            filtered_imag[:, c] = i.squeeze(-1)

        # Combined output with learned scaling
        return torch.stack(
            [
                (self.scale[..., 0] * filtered_real),
                (self.scale[..., 1] * filtered_imag),
            ],
            dim=-1,
        )

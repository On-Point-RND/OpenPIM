"""
Per-channel 1D amplitude LUT (simplest memoryless nonlinear core).

Each RF channel has its own table over |z_c|:
  z_out,c = z_c * g_c(|z_c|)

Uniform amplitude grid, linear interpolation, learnable complex gain per bin.
"""

import torch
import torch.nn as nn

from backbones.modules_filter import (
    TxaFilterEnsembleTorch,
    RxaFilterEnsembleTorch,
)


class LUT(nn.Module):
    """
    Memoryless per-channel LUT indexed by amplitude.

    Input / output: (B, T, C, 2)  — I/Q per RF channel.
    """

    def __init__(
        self,
        n_channels: int,
        n_bins: int = 64,
        amp_max: float = 1.0,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_bins = n_bins
        self.amp_max = amp_max

        gain = torch.zeros(n_channels, n_bins, 2)
        gain[..., 0] = 1.0
        self.gain = nn.Parameter(gain)

        bins = torch.linspace(0.0, amp_max, n_bins)
        self.register_buffer("bins", bins)

    def _interp_gain(self, mag: torch.Tensor) -> torch.Tensor:
        """mag (..., C) -> complex gain (..., C)."""
        *lead, channels = mag.shape
        flat_mag = mag.reshape(-1, channels)
        n_pts = flat_mag.shape[0]

        scale = (self.n_bins - 1) / self.amp_max
        idx = flat_mag.clamp(0.0, self.amp_max) * scale
        i0 = idx.long().clamp(0, self.n_bins - 1)
        i1 = (i0 + 1).clamp(0, self.n_bins - 1)
        w = (idx - i0.to(idx.dtype)).unsqueeze(-1)

        channel_ids = torch.arange(channels, device=mag.device)
        c_idx = channel_ids.unsqueeze(0).expand(n_pts, channels)
        g0 = self.gain[c_idx, i0]
        g1 = self.gain[c_idx, i1]
        g = g0 * (1.0 - w) + g1 * w

        gain_c = torch.complex(g[..., 0], g[..., 1])
        return gain_c.reshape(*lead, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.complex(x[..., 0], x[..., 1])
        mag = z.abs().clamp_min(0.0)
        z_out = z * self._interp_gain(mag)
        return torch.stack((z_out.real, z_out.imag), dim=-1)


class PerChannelLUT(nn.Module):

    def __init__(
        self,
        seq_len,
        tx_filt_size,
        rx_filt_size,
        n_channels,
        n_lut_bins: int = 64,
        lut_amp_max: float = 1.0,
    ):
        super().__init__()

        self.txa_filter_layers = TxaFilterEnsembleTorch(
            n_channels, tx_filt_size, seq_len
        )
        self.nlin_layer = LUT(
            n_channels,
            n_bins=n_lut_bins,
            amp_max=lut_amp_max,
        )
        self.rxa_filter_layers = RxaFilterEnsembleTorch(
            n_channels, rx_filt_size, seq_len
        )

    def forward(self, x, h_0=None):
        x = self.txa_filter_layers(x)
        x = self.nlin_layer(x)
        return self.rxa_filter_layers(x)


if __name__ == "__main__":
    C, N = 4, 32
    lut = LUT(C, n_bins=N, amp_max=1.0)
    mag = torch.rand(1, 16, C)
    phase = torch.rand(1, 16, C) * 2 * torch.pi
    x = torch.stack((mag * phase.cos(), mag * phase.sin()), dim=-1)
    y = lut(x)
    print("in:", tuple(x.shape), "out:", tuple(y.shape))
    print("gain table:", tuple(lut.gain.shape), "bins:", tuple(lut.bins.shape))

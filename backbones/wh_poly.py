import torch
import torch.nn as nn

from backbones.modules_filter import (
    TxaFilterEnsembleTorch,
    RxaFilterEnsembleTorch,
)

N_BASIS = 3


def _z_times(z: torch.Tensor, *f_of_mod: torch.Tensor) -> torch.Tensor:
    """φ_k(z) = z · f_k(|z|)."""
    return torch.stack([z * f for f in f_of_mod], dim=-1)


def _power_basis(z: torch.Tensor) -> torch.Tensor:
    x = z.abs()
    return _z_times(z, x, x.square(), x.pow(3))


class NlinCore(nn.Module):

    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.weight = nn.Parameter(
            torch.randn(n_channels, n_channels, N_BASIS)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.complex(x[..., 0], x[..., 1])
        phi = _power_basis(z)
        w = self.weight.to(phi.dtype)
        z_out = torch.einsum("btjk,cjk->btc", phi, w)
        return torch.stack((z_out.real, z_out.imag), dim=-1)


class MultiChannelRegression(nn.Module):
    def __init__(self, seq_len, tx_filt_size, rx_filt_size, n_channels):
        super().__init__()

        self.txa_filter_layers = TxaFilterEnsembleTorch(
            n_channels, tx_filt_size, seq_len
        )
        self.nlin_layer = NlinCore(n_channels)
        self.rxa_filter_layers = RxaFilterEnsembleTorch(
            n_channels, rx_filt_size, seq_len
        )

    def forward(self, x, h_0=None):
        x = self.txa_filter_layers(x)
        x = self.nlin_layer(x)
        return self.rxa_filter_layers(x)

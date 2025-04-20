import torch
from torch import nn


class GMP(nn.Module):
    def __init__(self, memory_length=11, degree=5, n_channels=1):
        super(GMP, self).__init__()
        self.memory_length = memory_length
        self.degree = degree
        self.n_channels = n_channels
        self.W = 1 + (degree - 1) * memory_length
        self.Weight = nn.Parameter(torch.Tensor(n_channels, self.W * memory_length))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Weight)

    def forward(self, x, h_0=None):
        B, N, C, _ = x.shape  # Input shape: (B, N, C, 2)
        x = x.permute(0, 2, 1, 3)  # Shape: (B, C, N, 2)
        x_real = x[..., 0]  # (B, C, N)
        x_imag = x[..., 1]  # (B, C, N)
        x_complex = torch.complex(x_real, x_imag)  # (B, C, N)

        # Pad with zeros
        pad = torch.zeros(B, C, self.memory_length - 1, device=x.device)
        padded_x = torch.cat([pad, x_complex], dim=2)  # (B, C, N + memory_length -1)

        # Create windows for x_complex
        windows_x = padded_x.unfold(
            2, self.memory_length, 1
        )  # (B, C, N, memory_length)

        # Compute amplitude and x_degree
        amp = torch.abs(padded_x)  # (B, C, N + memory_length -1)
        x_degree = torch.stack(
            [amp.pow(i) for i in range(1, self.degree)], dim=1
        )  # (B, degree-1, C, N + memory_length -1)

        # Create windows for x_degree
        windows_degree = x_degree.unfold(
            3, self.memory_length, 1
        )  # (B, degree-1, C, N, memory_length)

        # Expand windows_x to match degree dimension
        current_windows_x_expanded = windows_x.unsqueeze(1).expand(
            -1, self.degree - 1, -1, -1, -1
        )  # (B, degree-1, C, N, memory_length)

        # Multiply to get mul_term
        mul_term = (
            current_windows_x_expanded * windows_degree
        )  # (B, degree-1, C, N, memory_length)

        # Reshape mul_term to (B, C, N, (degree-1)*memory_length)
        mul_term = mul_term.permute(0, 2, 3, 1, 4).contiguous().view(B, C, N, -1)

        # Concatenate with windows_x to form x_input
        x_input = torch.cat([windows_x, mul_term], dim=3)  # (B, C, N, W*memory_length)

        # Multiply with Weight and sum
        product = x_input * self.Weight.unsqueeze(0).unsqueeze(
            2
        )  # (B, C, N, W*memory_length)
        complex_out = product.sum(dim=3)  # (B, C, N) complex

        # Prepare output
        complex_out = complex_out.permute(0, 2, 1)  # (B, N, C)
        out_real = complex_out.real  # (B, N, C)
        out_imag = complex_out.imag  # (B, N, C)
        out = torch.stack([out_real, out_imag], dim=3)  # (B, N, C, 2)

        return out

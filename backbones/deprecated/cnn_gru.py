import torch
import torch.nn as nn


class ActFn(nn.Module):
    def __init__(self):
        super().__init__()  # <---- This is required
        self.fn = torch.sin

    def forward(self, x):
        return self.fn(x) + x


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.fc_real = nn.Linear(in_features, out_features, bias=bias)
        self.fc_imag = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x_real, x_imag):
        real = self.fc_real(x_real) - self.fc_imag(x_imag)
        imag = self.fc_real(x_imag) + self.fc_imag(x_real)
        return real, imag


class CNN_GRU(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        n_channels,
        batch_size,
        hidden_size,
        batch_first=True,
        bias=True,
    ):
        super().__init__()

        # Input channels: 4 (I, Q, |x|, |x|², |x|³)
        # Input shape: (B, 4, 1, M1+M2+1) after view
        self.hidden_size = hidden_size
        self.n_channels = n_channels
        self.input_size = input_size

        self.filter_cnn_in = nn.Sequential(
            nn.Conv1d(
                in_channels=n_channels * 4,
                out_channels=n_channels * 2,
                kernel_size=3,
                padding=1,
            ),
            ActFn(),
        )

        # self.linear = nn.Linear(
        #     input_size * 4 * n_channels, input_size * n_channels, bias=bias
        # )

        # LSTM input size
        self.lstm = nn.GRU(
            input_size=n_channels * 2,
            hidden_size=n_channels * 2,
            batch_first=batch_first,
        )

        # self.fir = nn.Linear(
        #     input_size * hidden_size, n_channels * output_size, bias=bias
        # )

    def forward(self, x, h0):

        B, N, C, T = x.shape
        # Feature Extraction
        i_x = x[..., 0]
        q_x = x[..., 1]
        amp2 = torch.pow(i_x, 2) + torch.pow(q_x, 2)

        x = torch.cat(
            (
                i_x,
                q_x,
                i_x * amp2,
                q_x * amp2,
            ),
            dim=2,
        ).view(B, -1, N)

        filtered = self.filter_cnn_in(x)

        # linear_our = self.linear(filtered.view(B, -1)).view(B, -1, self.n_channels)

        h0 = torch.zeros(1, B, self.n_channels * 2).to(x.device)
        # c0 = torch.zeros(1, B, self.n_channels * 2).to(x.device)
        # LSTM processing
        _, h_out = self.lstm(filtered.view(B, N, -1), h0)  # -> (B, N, lstm_hidden)

        # out = self.fir(lstm_out.reshape(B, -1))

        return h_out.view(B, self.n_channels, 2)


class CNN_ONLY(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        n_channels,
        batch_size,
        hidden_size,
        batch_first=True,
        bias=True,
    ):
        super().__init__()

        # Input channels: 4 (I, Q, |x|, |x|², |x|³)
        # Input shape: (B, 4, 1, M1+M2+1) after view
        self.hidden_size = hidden_size
        self.n_channels = n_channels
        self.input_size = input_size

        self.filter_cnn_in = nn.Sequential(
            nn.Conv1d(
                in_channels=n_channels * 4,
                out_channels=n_channels * 2,
                kernel_size=3,
                padding=1,
            ),
            ActFn(),
        )

        self.linear = ComplexLinear(input_size, 1, bias=bias)

    def forward(self, x, h0):

        B, N, C, T = x.shape
        # Feature Extraction
        i_x = x[..., 0]
        q_x = x[..., 1]
        amp2 = torch.pow(i_x, 2) + torch.pow(q_x, 2)

        x = torch.cat(
            (
                i_x,
                q_x,
                i_x * amp2,
                q_x * amp2,
            ),
            dim=2,
        ).view(B, -1, N)

        filtered = self.filter_cnn_in(x).view(B * C, -1, 2)
        x_i, x_q = self.linear(filtered[:, :, 0], filtered[:, :, 1])
        output = torch.stack([x_i, x_q], dim=-1)
        return output.view(B, self.n_channels, 2)

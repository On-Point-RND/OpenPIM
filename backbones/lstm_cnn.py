import torch
import torch.nn as nn


class CNN_LSTM_FIR(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        n_channels,
        batch_size,
        hidden_size,
        batch_first,
        bias,
    ):
        super().__init__()

        # Input channels: 4 (I, Q, |x|, |x|², |x|³)
        # Input shape: (B, 4, 1, M1+M2+1) after view
        self.hidden_size = hidden_size
        self.n_channels = n_channels
        self.input_size = input_size
        self.c = 12
        self.f = 6
        self.act = nn.Tanh()

        self.filter_cnn_in = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=self.c,
                kernel_size=(1, self.f),
                padding=(0, 0),
            ),
            # NOTE: NON LINEARITY IS HERE
            self.act,
        )

        self.linear = nn.Linear(
            self.c,
            1,
            bias=bias,
        )

        # LSTM input size
        self.lstm = nn.LSTM(
            input_size=self.n_channels * self.f,
            hidden_size=hidden_size,  # self.n_channels * self.c * 6,
            batch_first=batch_first,
        )

        self.fir = nn.Linear(
            (input_size - 5) * hidden_size, n_channels * output_size, bias=bias
        )

        self.amp_weight = nn.Parameter(torch.tensor(1.0))
        self.amp2_weight = nn.Parameter(torch.tensor(1.0))

        self.scale = nn.Parameter(torch.ones(1, self.n_channels))  # Shape: (1, C, 2)

    def forward(self, x, h0):
        B, N, C, T = x.shape
        # Feature Extraction
        i_x = x[..., 0]
        q_x = x[..., 1]
        amp2 = torch.pow(i_x, 2) + torch.pow(q_x, 2)
        amp = torch.sqrt(amp2)
        cos = i_x / amp
        sin = q_x / amp

        # B, C, LEN
        x = torch.cat(
            (
                i_x,
                q_x,
                self.amp_weight * amp,
                self.amp2_weight * amp2,
                cos,
                sin,
            ),
            dim=2,
        ).view(B, C, 6, N)

        filtered = self.act(
            self.filter_cnn_in(x).view(B, self.f * C * (N + 1 - self.f), self.c)
        )

        linear_out = self.linear(filtered).view(B, -1, self.n_channels * self.f)
        # (B, C,1, input_size)

        h0 = torch.zeros(1, B, self.hidden_size).to(x.device)
        c0 = torch.zeros(1, B, self.hidden_size).to(x.device)
        # LSTM processing
        lstm_out, _ = self.lstm(linear_out, (h0, c0))  # -> (B, N, lstm_hidden)

        # FIR filtering
        out = self.fir(lstm_out.reshape(B, -1))

        return self.scale * out.view(B, self.n_channels, 2)

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

        self.filter_cnn_in = nn.ModuleList()
        for i in range(0, n_channels):
            self.filter_cnn_in.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=1,
                        kernel_size=(3, 3),
                        padding=(1, 1),
                    ),
                    nn.Tanh(),
                )
            )

        self.linear = nn.Linear(
            input_size * 4 * n_channels, input_size * n_channels, bias=bias
        )

        # LSTM input size
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            batch_first=batch_first,
        )

        self.fir = nn.Linear(
            input_size * hidden_size, n_channels * output_size, bias=bias
        )

    def forward(self, x, h0):
        B, N, C, T = x.shape
        # Feature Extraction
        i_x = x[..., 0]
        q_x = x[..., 1]
        amp2 = torch.pow(i_x, 2) + torch.pow(q_x, 2)
        amp = torch.sqrt(amp2)
        amp3 = torch.pow(amp, 3)
        # cos = i_x / amp
        # sin = q_x / amp

        # B, C, LEN
        x = torch.cat(
            (
                i_x.view(B, C, 1, -1),
                q_x.view(B, C, 1, -1),
                amp.view(B, C, 1, -1),
                amp3.view(B, C, 1, -1),
            ),
            dim=2,
        )

        filtered = torch.zeros(
            (B, 4, self.n_channels, self.input_size), device=x.device
        )

        for c, filt_layer in enumerate(self.filter_cnn_in):
            tr = x[:, c, ...].view(B, 1, 4, -1)
            filtered[:, :, c, ...] = filt_layer(tr).squeeze(1)  # (B, C,1, input_size)

        linear_our = self.linear(filtered.view(B, -1)).view(B, -1, self.n_channels)

        h0 = torch.zeros(1, B, self.hidden_size).to(x.device)
        c0 = torch.zeros(1, B, self.hidden_size).to(x.device)
        # LSTM processing
        lstm_out, _ = self.lstm(linear_our, (h0, c0))  # -> (B, N, lstm_hidden)

        # lstm_out ()

        # FIR filtering
        out = self.fir(lstm_out.reshape(B, -1))

        return out.view(B, self.n_channels, 2)

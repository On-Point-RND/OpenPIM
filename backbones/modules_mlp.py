import torch.nn as nn
import torch.nn.init as init
import torch


class SingleLayerPerceptron(nn.Module):
    def __init__(
        self,
        n_channels,
        nonlinearity="gelu",
        input_size=32,
        output_size=32
    ):
        super().__init__()
        self.n_channels = n_channels

        # Linear layer: input and output are both 2 * n_channels
        self.linear = nn.Linear(input_size, output_size, bias=True)

        # Initialize weights as identity matrix
        self._initialize_as_identity()

        # Set non-linearity
        self.nlin = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "silu": nn.SiLU(),
            "gelu": nn.GELU(),
            "none": nn.Identity(),
        }[nonlinearity]

    def _initialize_as_identity(self):
        init.eye_(self.linear.weight)
        # Optional: zero out the bias
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, x):
        transformed = self.linear(x)
        return self.nlin(transformed)


class SingleChannelPerceptron(nn.Module):
    def __init__(
        self,
        output_size=2,
        hidden_size=16,
        num_layers=2,
        activation="silu"
    ):
        super().__init__()
        layers = []
        activation_map = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "silu": nn.SiLU(),
            "gelu": nn.GELU(),
            "none": nn.Identity(),
            "selu": nn.SELU(),
            "softplus": nn.Softplus(),
        }

        # Input: [I, Q, |x|] (3 features)
        for i in range(num_layers):
            in_dim = 3 if i == 0 else hidden_size
            out_dim = output_size if i == num_layers - 1 else hidden_size
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(activation_map[activation])
        self.net = nn.Sequential(*layers)

        # Initialize to identity: f(x) ≈ x
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.fill_(1.0)

    def forward(self, x):
        amps = torch.norm(x, dim=-1, keepdim=True)
        # Features: I, Q, amplitude
        features = torch.cat([x, amps], dim=-1)
        return self.net(features)

import torch
import torch.nn as nn
import torch.nn.init as init

from backbones.common_modules import (
    TxaFilterEnsembleTorch,
    RxaFilterEnsembleTorch,
)


class SimpleFunction(nn.Module):
    def __init__(self, function_type="abs"):
        super().__init__()
        self.function_type = function_type
        
        if function_type not in ["abs", "modulus", "sin", "cos"]:
            raise ValueError(f"function_type must be one of: 'abs', 'modulus', 'sin', 'cos', got '{function_type}'")

    def forward(self, x):
        """
        Apply simple function to input tensor of shape 
        (batch, seq_len, n_channels, 2)
        Args:
            x: Input tensor of shape (batch, seq_len, n_channels, 2)
        Returns:
            Tensor of shape (batch, seq_len, n_channels, 1)
        """
        re, im = x[..., 0], x[..., 1]
        if self.function_type == "abs":
            result = torch.abs(re) + torch.abs(im)
        elif self.function_type == "modulus":
            result = torch.sqrt(re ** 2 + im ** 2)
        elif self.function_type == "sin":
            modulus = torch.sqrt(re ** 2 + im ** 2)
            result = torch.sin(modulus)
        elif self.function_type == "cos":
            modulus = torch.sqrt(re ** 2 + im ** 2)
            result = torch.cos(modulus)
        else:
            raise ValueError(f"Unknown function_type: {self.function_type}")

        return result.unsqueeze(-1)


class SimpleMLP(nn.Module):
    def __init__(self, hidden_size=16, num_layers=2, activation="silu"):
        super().__init__()
        layers = []
        # Input: [I, Q, |x|] (3 features)
        # Activation function mapping
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
        for i in range(num_layers):
            in_dim = 3 if i == 0 else hidden_size
            out_dim = 1 if i == num_layers - 1 else hidden_size
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


class FeatureGeneratorMoe(nn.Module):
    def __init__(self, hidden_size=16, num_layers=2):
        super().__init__()

        self.activations = [
            "relu", "tanh", "elu", "silu",
            "gelu", "none", "selu", "softplus"
        ]

        self.simple_functions = [
            "abs", "modulus", "sin", "cos"
        ]

        self.num_experts = len(self.activations) + len(self.simple_functions)

        self.experts = nn.ModuleList([
            SimpleMLP(hidden_size, num_layers, activation)
            for activation in self.activations
        ] + [
            SimpleFunction(function_type=func)
            for func in self.simple_functions
        ])
        self.expert_weights = nn.Parameter(torch.ones(self.num_experts) / self.num_experts)

    def forward(self, x):
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=-1)
        output = torch.sum(expert_outputs * self.expert_weights, dim=-1, keepdim=True)
        return output


class EnrichedPerceptron(nn.Module):
    def __init__(self, n_channels, nonlinearity):
        super().__init__()
        self.n_channels = n_channels
        self.linear = nn.Linear(3 * n_channels, 2 * n_channels, bias=True)
        self._initialize_as_identity()
        self.enrich_layer = FeatureGeneratorMoe()
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
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, x):
        batch, time, n_ch, _ = x.shape
        x_flat = x.view(batch * time * n_ch, -1)
        enrichment = self.enrich_layer(x_flat)
        enrichment = enrichment.view(batch * time, n_ch, 1)
        x_flat = x_flat.view(batch * time, n_ch, 2)
        enriched_input = torch.cat([x_flat, enrichment], dim=-1)
        enriched_input = enriched_input.view(batch * time, -1)
        transformed = self.linear(enriched_input)
        transformed = transformed.view(batch, time, n_ch, 2)
        return self.nlin(transformed)


class SingleLayerPerceptron(nn.Module):
    def __init__(self, n_channels, nonlinearity):
        super().__init__()
        self.n_channels = n_channels
        # Linear layer: input and output are both 2 * n_channels
        self.linear = nn.Linear(2 * n_channels, 2 * n_channels, bias=True)
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
        batch, time = x.shape[0], x.shape[1]
        x_flat = x.view(batch * time, -1)
        transformed = self.linear(x_flat)
        transformed = transformed.view(batch, time, self.n_channels, 2)
        return self.nlin(transformed)


class NlinCore(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        nonlinearity = "silu"
        num_layers = 3
        layers = []
        layers.append(EnrichedPerceptron(n_channels, nonlinearity))
        for _ in range(num_layers - 1):
            layers.append(SingleLayerPerceptron(n_channels, nonlinearity))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MoeEnriched(nn.Module):
    def __init__(self, in_seq_size, out_seq_size, n_channels):
        super().__init__()
        self.n_channels = n_channels

        self.txa_filter_layers = TxaFilterEnsembleTorch(
            n_channels, in_seq_size, out_seq_size
        )

        self.nlin_layer = NlinCore(n_channels)

        self.rxa_filter_layers = RxaFilterEnsembleTorch(
            n_channels, out_seq_size
        )

        self.bn_output = nn.BatchNorm1d(n_channels)

    def forward(self, x, h_0=None):
        filtered_x = self.txa_filter_layers(x)
        nonlin_output = self.nlin_layer(filtered_x)
        filt_rxa = self.rxa_filter_layers(nonlin_output)
        output = self.bn_output(filt_rxa)
        return output

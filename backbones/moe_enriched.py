import torch
import torch.nn as nn
import torch.nn.init as init

from backbones.filter_modules import (
    TxaFilterEnsembleTorch,
    RxaFilterEnsembleTorch,
)

from backbones.mlp_modules import (
    SingleLayerPerceptron,
    SingleChannelPerceptron,
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
            SingleChannelPerceptron(output_size=1, activation=activation)
            for activation in self.activations
        ] + [
            SimpleFunction(function_type=func)
            for func in self.simple_functions
        ])
        self.expert_weights = nn.Parameter(torch.ones(self.num_experts) / self.num_experts)

    def forward(self, x):
        # Input: (batch_time * n_ch, 2)
        expert_outputs = []
        for expert in self.experts:
            # Each expert processes (batch_time * n_ch, 2) and returns
            # (batch_time * n_ch, 1)
            expert_outputs.append(expert(x))
        # expert_outputs stacked: (batch_time * n_ch, 1, num_experts)
        expert_outputs = torch.stack(expert_outputs, dim=-1)
        # output: (batch_time * n_ch, 1)
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
        # Input: (batch * time_seq_len, n_ch, 2)
        batch_time, n_ch, _ = x.shape
        # x_flat: (batch_time * n_ch, 2)
        x_flat = x.view(batch_time * n_ch, 2)
        # enrichment: (batch_time * n_ch, 1)
        enrichment = self.enrich_layer(x_flat)
        # enrichment: (batch_time, n_ch, 1)
        enrichment = enrichment.view(batch_time, n_ch, 1)

        # Concatenate [I, Q, enrichment]:
        # (batch * time_seq_len, n_ch, 3)
        enriched_input = torch.cat([x, enrichment], dim=-1)

        # Flatten for linear layer:
        # (batch * time_seq_len, n_ch * 3)
        enriched_input_flat = enriched_input.view(batch_time, -1)
        # transformed: (batch_time, 2 * n_ch)
        transformed = self.linear(enriched_input_flat)
        # return: (batch_time, 2 * n_ch)
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
            layers.append(SingleLayerPerceptron(
                n_channels, 
                nonlinearity,
                input_size=2 * n_channels,
                output_size=2 * n_channels
            ))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Input: (batch, time_seq_len, n_ch, 2)
        batch, time_seq_len = x.shape[0], x.shape[1]
        n_ch = self.n_channels
        # Flatten for processing:
        # (batch * time_seq_len, n_ch, 2)
        x_flat = x.view(batch * time_seq_len, n_ch, 2)
        transformed = self.model(x_flat)
        transformed = transformed.view(batch, time_seq_len, n_ch, 2)
        return transformed


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

    def forward(self, x, h_0=None):
        filtered_x = self.txa_filter_layers(x)
        nonlin_output = self.nlin_layer(filtered_x)
        filt_rxa = self.rxa_filter_layers(nonlin_output)
        return filt_rxa

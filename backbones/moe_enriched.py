import torch
import torch.nn as nn
import torch.nn.init as init

from backbones.modules_filter import (
    TxaFilterEnsembleTorch,
    RxaFilterEnsembleTorch,
)

from backbones.modules_mlp import (
    SingleLayerPerceptron,
    SingleChannelPerceptron,
)


_SIMPLE_FUNCTION_TYPES = [
    "abs", "modulus", "sin", "cos",
    "phase", "power", "log1p", "tanhz",
]

EXPERT_MODES = ("simple_only", "trainable_only", "both")


class SimpleFunction(nn.Module):
    def __init__(self, function_type="abs"):
        super().__init__()
        self.function_type = function_type
        if function_type not in _SIMPLE_FUNCTION_TYPES:
            raise ValueError(
                f"function_type must be one of {_SIMPLE_FUNCTION_TYPES}, got '{function_type}'"
            )

    def forward(self, x):
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
        elif self.function_type == "phase":
            result = torch.atan2(im, re)
        elif self.function_type == "power":
            result = re ** 2 + im ** 2
        elif self.function_type == "log1p":
            modulus = torch.sqrt(re ** 2 + im ** 2)
            result = torch.log1p(modulus)
        elif self.function_type == f"tanhz":
            modulus = torch.sqrt(re ** 2 + im ** 2)
            result = torch.tanh(modulus)
        else:
            raise ValueError(f"Unknown function_type: {self.function_type}")
        return result.unsqueeze(-1)


class FeatureGeneratorMoe(nn.Module):
    def __init__(self, hidden_size=16, num_layers=2, expert_mode="both"):
        super().__init__()
        if expert_mode not in EXPERT_MODES:
            raise ValueError(f"expert_mode must be one of {EXPERT_MODES}, got '{expert_mode}'")
        self.expert_mode = expert_mode

        self.activations = [
            "relu", "tanh", "elu", "silu",
            "gelu", "none", "selu", "softplus"
        ]
        self.simple_functions = _SIMPLE_FUNCTION_TYPES.copy()

        expert_names = []
        experts = []
        if expert_mode in ("trainable_only", "both"):
            for a in self.activations:
                experts.append(
                    SingleChannelPerceptron(output_size=1, activation=a)
                )
                expert_names.append(a)
        if expert_mode in ("simple_only", "both"):
            for f in self.simple_functions:
                experts.append(SimpleFunction(function_type=f))
                expert_names.append(f)
        self.num_experts = len(experts)
        self.experts = nn.ModuleList(experts)
        self.expert_names = expert_names
        self.expert_weights = nn.Parameter(
            torch.ones(self.num_experts) / self.num_experts
        )

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
        output = torch.sum(
            expert_outputs * self.expert_weights, dim=-1, keepdim=True
        )
        return output


class EnrichedPerceptron(nn.Module):
    def __init__(self, n_channels, nonlinearity, expert_mode="both"):
        super().__init__()
        self.n_channels = n_channels
        self.linear = nn.Linear(3 * n_channels, 2 * n_channels, bias=True)
        self._initialize_as_identity()
        self.enrich_layer = FeatureGeneratorMoe(expert_mode=expert_mode)
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
        # x is expected to be (batch * time, n_ch, 2)
        batch_time, n_ch, _ = x.shape
        x_flat = x.view(batch_time * n_ch, 2)
        enrichment = self.enrich_layer(x_flat)
        enrichment = enrichment.view(batch_time, n_ch, 1)

        enriched_input = torch.empty(
            batch_time, n_ch, 3,
            dtype=x.dtype, device=x.device
        )
        enriched_input[:, :, :2] = x
        enriched_input[:, :, 2:] = enrichment

        # Flatten for linear layer: (batch * time, n_ch * 3)
        enriched_input_flat = enriched_input.view(batch_time, -1)
        transformed = self.linear(enriched_input_flat)
        return self.nlin(transformed)


class NlinCore(nn.Module):
    def __init__(self, n_channels, expert_mode="both"):
        super().__init__()
        self.n_channels = n_channels
        nonlinearity = "silu"
        num_layers = 3
        layers = []
        layers.append(
            EnrichedPerceptron(
                n_channels, nonlinearity, expert_mode=expert_mode
            )
        )
        for _ in range(num_layers - 1):
            layers.append(SingleLayerPerceptron(
                n_channels, 
                nonlinearity,
                input_size=2 * n_channels,
                output_size=2 * n_channels
            ))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        batch, time_seq_len = x.shape[0], x.shape[1]
        n_ch = self.n_channels
        # Flatten for processing: (batch * time_seq_len, n_ch, 2)
        x_flat = x.view(batch * time_seq_len, n_ch, 2)
        transformed = self.model(x_flat)
        # Reshape back: (batch, time_seq_len, n_channels, 2)
        transformed = transformed.view(batch, time_seq_len, n_ch, 2)
        return transformed


class MoeEnriched(nn.Module):
    """Experts: trainable (SingleChannelPerceptron) + simple (SimpleFunction).
    expert_mode: one of 'simple_only', 'trainable_only', 'both'.
    """

    def __init__(
        self,
        seq_len,
        tx_filt_size, rx_filt_size,
        n_channels,
        expert_mode="both"
    ):
        super().__init__()
        if expert_mode not in EXPERT_MODES:
            raise ValueError(f"expert_mode must be one of {EXPERT_MODES}, got '{expert_mode}'")
        self.n_channels = n_channels
        self.expert_mode = expert_mode

        self.txa_filter_layers = TxaFilterEnsembleTorch(
            n_channels, tx_filt_size, seq_len
        )

        self.nlin_layer = NlinCore(
            n_channels, expert_mode=expert_mode
        )

        self.rxa_filter_layers = RxaFilterEnsembleTorch(
            n_channels, rx_filt_size, seq_len
        )

    def forward(self, x, h_0=None):
        filtered_x = self.txa_filter_layers(x)
        nonlin_output = self.nlin_layer(filtered_x)
        return self.rxa_filter_layers(nonlin_output)

    def get_expert_weights(self):
        weights = self.nlin_layer.model[0].enrich_layer.expert_weights
        return weights.detach().cpu()

    def get_expert_names(self):
        return self.nlin_layer.model[0].enrich_layer.expert_names

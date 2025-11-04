import torch
import torch.nn as nn
import torch.nn.init as init

from backbones.common_modules import (
    TxaFilterEnsembleTorch, RxaFilterEnsembleTorch
)


def keep_top_k(x, k):
    # x shape: (n_batch, n)
    _, n = x.shape
    if n < k:
        return x
    # Get k-th largest values for each batch
    kth_values = torch.kthvalue(x, n - k, dim=1).values  # shape: (n_batch,)

    # Expand kth_values to match shape of x
    kth_values = kth_values.unsqueeze(1)  # shape: (n_batch, 1)

    # Create mask and apply it
    mask = x > kth_values
    result = torch.where(mask, x, torch.tensor(float('-inf'), device=x.device))

    return result


class Gate(nn.Module):
    def __init__(self, n_seq, n_channels, n_experts, k=2, aux_loss_weight=1e-6):
        super().__init__()
        self.n_seq = n_seq
        self.n_channels = n_channels
        self.n_experts = n_experts
        self.main_gate = nn.Linear(n_channels * n_seq * 2, n_experts)
        self.gate_activation = nn.Softmax(dim=1)
        self.k = k
        self.aux_loss_weight = aux_loss_weight

    def forward(self, x, return_aux_loss=False):
        n_batch, *_ = x.shape
        x_reshaped = x.reshape(n_batch, -1)
        gate_out = self.main_gate(x_reshaped)
        gate_top_k = keep_top_k(gate_out, self.k)
        gate_out = self.gate_activation(gate_top_k)

        if return_aux_loss:
            # Sum of gate weights for each expert across batch
            expert_sums = torch.sum(gate_out, dim=0)  # shape: [n_experts]

            # Mean and std of the expert sums
            mean_sum = torch.mean(expert_sums)
            std_sum = torch.std(expert_sums, unbiased=True)

            # Coefficient of variation squared
            cv_squared = (std_sum / (mean_sum + 1e-6)) ** 2
            # Scaling
            aux_loss = self.aux_loss_weight * cv_squared
            return gate_out, aux_loss
        return gate_out


class SingleLayerPerceptron(nn.Module):
    def __init__(self, n_channels, nonlinearity="silu"):
        super().__init__()
        # Store number of channels explicitly
        self.n_channels = n_channels

        # Initialize matrices as identity matrices
        self.linear = nn.Linear(2 * n_channels, 2 * n_channels, bias=True)

        # Initialize weights as identity matrix
        self._initialize_as_identity()

        # Set non-linearity
        self.nlin = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "silu": nn.SiLU(),
            "none": nn.Identity(),
            "gelu": nn.GELU(),
            "selu": nn.SELU(),
            "softplus": nn.Softplus(),
        }[nonlinearity]

    def _initialize_as_identity(self):
        # Identity matrix initialization for weight
        assert (
            self.linear.weight.shape[0] == self.linear.weight.shape[1]
        ), "Weight matrix must be square"
        init.eye_(self.linear.weight)
        # Zero out the bias
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, x):
        batch, time = x.shape[0], x.shape[1]
        x_flat = x.view(batch * time, -1)  # Shape: (B*T, C*2)
        transformed = self.linear(x_flat)
        transformed = transformed.view(batch, time, self.n_channels, 2)
        return self.nlin(transformed)


class NlinCore(nn.Module):
    def __init__(self, n_channels, num_layers=5, nonlinearity="silu"):
        super().__init__()
        self.n_channels = n_channels
        layers = []
        for _ in range(num_layers):
            layers.append(SingleLayerPerceptron(n_channels, nonlinearity))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MoeLayer(nn.Module):
    def __init__(self, seq_size, n_channels, return_aux_loss=True, aux_loss_weight=1e-6):
        super().__init__()
        self.n_channels = n_channels
        self.seq_size = seq_size
        self.return_aux_loss = return_aux_loss
        self.aux_loss_weight = aux_loss_weight

        experts_list = [
            "relu", "tanh", "elu", "silu", "none",
            "gelu", "selu", "softplus"
        ]

        n_experts = len(experts_list)
        self.experts = nn.ModuleList([
            NlinCore(n_channels, nonlinearity=expert)
            for expert in experts_list
        ])
        self.gate = Gate(
            seq_size,
            n_channels,
            n_experts,
            k=3,
            aux_loss_weight=aux_loss_weight,
        )

    def forward(self, x, h_0=None):
        if self.return_aux_loss:
            gate_out, aux_loss = self.gate(
                x, return_aux_loss=self.return_aux_loss
            )
        else:
            gate_out = self.gate(x)
        experts_outputs = torch.stack(
            [expert(x) for expert in self.experts]
        )
        weighted_output = torch.einsum('be,eb...->b...', gate_out, experts_outputs)
        if self.return_aux_loss:
            return weighted_output, aux_loss
        return weighted_output


class MixtureMultiMLP(nn.Module):
    def __init__(self, in_seq_size, out_seq_size, n_channels,
                 return_aux_loss=True, aux_loss_weight=1e-6):
        super().__init__()
        self.out_seq_size = out_seq_size
        self.n_channels = n_channels
        self.return_aux_loss = return_aux_loss
        self.moe_layer = MoeLayer(
            out_seq_size,
            n_channels,
            return_aux_loss=return_aux_loss,
            aux_loss_weight=aux_loss_weight,
        )

        self.txa_filter_layers = TxaFilterEnsembleTorch(
            n_channels, in_seq_size, out_seq_size
        )

        self.rxa_filter_layers = RxaFilterEnsembleTorch(
            n_channels, out_seq_size
        )

        self.bn_output = nn.BatchNorm1d(n_channels)

    def forward(self, x, h_0=None):
        filtered_x = self.txa_filter_layers(x)
        if self.return_aux_loss:
            distorted_x, aux_loss = self.moe_layer(filtered_x)
        else:
            distorted_x = self.moe_layer(filtered_x)
        filt_rxa = self.rxa_filter_layers(distorted_x)
        output = self.bn_output(filt_rxa)
        if self.return_aux_loss:
            return output, aux_loss
        return output

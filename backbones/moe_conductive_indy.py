import torch
import torch.nn as nn

from backbones.common_modules import (
    TxaFilterEnsembleTorch,
    RxaFilterEnsembleTorch,
)


def keep_top_k(x, k):
    # x shape: (n_batch, n)
    _, n = x.shape
    if n < k:
        return x
    # Get k-th largest values for each batch
    kth_values = torch.kthvalue(x, n - k, dim=1).values  # shape: (n_batch,)
    
    # Expand kth_values to match x shape for comparison
    kth_values = kth_values.unsqueeze(1)  # shape: (n_batch, 1)
    
    # Create mask and apply it
    mask = x > kth_values
    result = torch.where(mask, x, torch.tensor(float('-inf'), device=x.device))
    
    return result


class Gate(nn.Module):
    def __init__(self, n_seq, n_experts, k=2):
        super().__init__()
        self.n_seq = n_seq
        self.n_experts = n_experts
        self.main_gate = nn.Linear(2 * n_seq, n_experts)
        self.noise_gate = nn.Linear(2 * n_seq, n_experts)
        self.gate_activation = nn.Softmax(dim=1)
        self.noise_activation = nn.Softplus()
        self.k = k

    def forward(self, x):
        n_batch, n_seq, n_channels, _ = x.shape
        x_reshaped = x.swapaxes(1, 2).reshape(n_batch * n_channels, 2 * n_seq)
        main_gate_out = self.main_gate(x_reshaped)
        noise_gate_out = self.noise_activation(self.noise_gate(x_reshaped))
        gate_out = main_gate_out + noise_gate_out
        gate_top_k = keep_top_k(gate_out, self.k)
        gate_out = self.gate_activation(gate_top_k)
        return gate_out


class MoELayer(nn.Module):
    def __init__(self, n_seq, k=2):
        super().__init__()
        self.n_seq = n_seq
        experts_nlin = [
            "relu", "tanh", "elu", "silu", "none",
            "gelu", "selu", "softplus"
        ]
        self.experts = nn.ModuleList([
            PhaseAwareNonlin(nonlinearity=expert)
            for expert in experts_nlin
        ])
        self.gate = Gate(n_seq, len(experts_nlin), k)

    def forward(self, x):
        gate_out = self.gate(x)
        experts_outputs = torch.stack(
            [expert(x) for expert in self.experts]
        )
        weighted_output = compute_moe_output(experts_outputs, gate_out)
        return weighted_output


def compute_moe_output(expert_outputs, moe_gate):
    # expert_outputs: (8, batch, seq, channels, 2)
    # moe_gate: (batch, channels, n_experts)
    
    # Reshape expert outputs to combine batch and channels
    # (8, batch, seq, channels, 2) -> (8, batch*channels, seq, 2)
    batch, channels = expert_outputs.shape[1], expert_outputs.shape[3]
    expert_outputs_reshaped = expert_outputs.reshape(8, batch*channels, -1, 2)
    
    # Reshape moe gate to match
    # (batch, channels, n_experts) -> (batch*channels, n_experts)
    # moe_gate_reshaped = moe_gate.reshape(batch*channels, -1)
    
    # Compute the weighted sum using einsum
    output = torch.einsum('be,ebst->bst', 
                         moe_gate, 
                         expert_outputs_reshaped)
    
    # Reshape back to original format
    # (batch*channels, seq, 2) -> (batch, seq, channels, 2)
    output = output.reshape(batch, channels, -1, 2).transpose(1, 2)
    
    return output


class PhaseAwareNonlin(nn.Module):
    def __init__(self, hidden_size=16, num_layers=2, nonlinearity="silu"):
        super().__init__()
        layers = []
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
        # Input: [I, Q, |x|] (3 features)
        for i in range(num_layers):
            in_dim = 3 if i == 0 else hidden_size
            out_dim = 2 if i == num_layers - 1 else hidden_size
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(self.nlin)
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


class MoEConductiveIndyE(nn.Module):
    def __init__(self, in_seq_size, out_seq_size, n_channels):
        super().__init__()
        self.out_seq_size = out_seq_size
        self.n_channels = n_channels
        self.moe_layer = MoELayer(out_seq_size, k=3)
        self.txa_filter_layers = TxaFilterEnsembleTorch(
            n_channels, in_seq_size, out_seq_size
        )

        self.rxa_filter_layers = RxaFilterEnsembleTorch(n_channels, out_seq_size)

        self.bn_output = nn.BatchNorm1d(n_channels)  # For complex output

    def forward(self, x, h_0=None):
        filtered_x = self.txa_filter_layers(x)
        weighted_output = self.moe_layer(filtered_x)
        filt_rxa = self.rxa_filter_layers(weighted_output)
        output = self.bn_output(filt_rxa)
        return output

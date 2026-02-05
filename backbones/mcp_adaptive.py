import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.modules_filter import (
    TxaFilterEnsembleTorch,
    RxaFilterEnsembleTorch,
)


class SingleLayerPerceptron(nn.Module):
    def __init__(self, n_channels, nonlinearity, input_size=32, output_size=32):
        super().__init__()
        self.n_channels = n_channels
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size, bias=True)
        #self._initialize_as_identity()

        # Set non-linearity
        self.nlin = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "silu": nn.SiLU(),
            "gelu": nn.GELU(),
            "none": nn.Identity(),
        }[nonlinearity]

    # def _initialize_as_identity(self):
    #     init.eye_(self.linear.weight)
    #     # Optional: zero out the bias
    #     if self.linear.bias is not None:
    #         init.zeros_(self.linear.bias)

    def forward(self, x):
        batch, time = x.shape[0], x.shape[1]
        x_flat = x.view(batch * time, -1)
        transformed = self.linear(x_flat)
        transformed = transformed.view(batch, time, self.output_size//2, 2)
        return self.nlin(transformed)


def stable_softmax(logits, eps=1e-2):
    # Standard numerical stability trick: subtract max
    logits_max = torch.max(logits, dim=0, keepdim=True)[0]
    exps = torch.exp(logits - logits_max)
    # Add epsilon to the denominator to prevent division by zero
    return exps / (torch.sum(exps, dim=0, keepdim=True) + eps)






class NlinCore(nn.Module):
    def __init__(self, n_channels, num_layers=4, nonlinearity="silu", compr = 32, device = 'cuda:0'):
        super().__init__()
        self.n_channels = n_channels
        self.tails = nn.ModuleList()

        all_comprs = torch.tensor([2, 8, 16, 32], device=device)

        self.all_comprs = all_comprs
        self.n_comp_models = all_comprs.shape[0]
        # Weights for combining component models via dot product
        weights = torch.ones(self.n_comp_models, device=device)
        self.lambdas = nn.Parameter(weights)
        
        # Shared head: layers 0-1 with fixed size 32
        head_layers = []
        for i in range(2):
            head_layers.append(
                SingleLayerPerceptron(
                    n_channels,
                    nonlinearity,
                    input_size=32,
                    output_size=32
                )
            )
        self.head = nn.Sequential(*head_layers)
        
        # Variable tails: layers 2-3 with compression-specific sizes
        for i, compr in enumerate(all_comprs):
            tail_layers = []
            # if i == 0:
            #     tail_layers.append(nn.Identity())
            # else:
            for layer_idx in range(2, 4):
                tail_layers.append(
                    SingleLayerPerceptron(
                        n_channels,
                        nonlinearity,
                        input_size=compr if layer_idx == 3 else 32,
                        output_size=compr if layer_idx == 2 else 32
                    )
                )
            self.tails.append(nn.Sequential(*tail_layers))

    def forward(self, x):
        # 2. Apply Softmax ONCE here
        #safe_lambdas = torch.clamp(self.lambdas, min=0.1, max=5.0)
        weights = F.softmax(self.lambdas) 
        #weights = 0.95 * weights + 0.05 * (1.0 / self.n_comp_models) 
        head_output = self.head(x)
        
        # 3. Avoid pre-allocating large empty tensors if possible, 
        # or ensure they are zeroed to avoid garbage memory NaNs
        outputs = []
        for tail in self.tails:
            outputs.append(tail(head_output).unsqueeze(-1))
        
        # Stack and multiply
        combined_output = torch.cat(outputs, dim=-1)
        return torch.matmul(combined_output, weights)



class McpAdaptive(nn.Module):
    def __init__(self, seq_len, tx_filt_size, rx_filt_size, n_channels):
        super().__init__()
        self.n_channels = n_channels

        self.txa_filter_layers = TxaFilterEnsembleTorch(
           n_channels, tx_filt_size, seq_len
        )

        self.nlin_layer = NlinCore(n_channels)

        self.rxa_filter_layers = RxaFilterEnsembleTorch(
            n_channels, rx_filt_size, seq_len
        )

        self.bn_output = nn.BatchNorm1d(n_channels)

    def forward(self, x, h_0=None):
        filtered_x = self.txa_filter_layers(x)
        nonlin_output = self.nlin_layer(filtered_x)
        filt_rxa = self.rxa_filter_layers(nonlin_output)
        output = self.bn_output(filt_rxa)
        return output
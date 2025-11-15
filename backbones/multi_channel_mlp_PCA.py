import torch.nn as nn
import torch.nn.init as init
import torch

import numpy as np
from sklearn.decomposition import PCA

from backbones.common_modules import (
    TxaFilterEnsembleTorch,
    RxaFilterEnsembleTorch,
)


class SingleLayerPerceptronPCA(nn.Module):
    def __init__(self, n_channels, nonlinearity="silu", n_pca_components=32, n_pca_batches=2):
        super().__init__()
        print(n_pca_components)
        self.n_channels = n_channels
        self.n_pca_components = n_pca_components
        self.n_pca_batches = n_pca_batches

        # Linear layer: input and output are both 2 * n_channels
        self.linear = nn.Linear(2 * n_channels, 2 * n_channels, bias=True)
        self._initialize_as_identity()

        # Non-linearity
        self.nlin = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "silu": nn.SiLU(),
            "none": nn.Identity(),
        }[nonlinearity]

        # PCA-related state
        self.pca = None
        self.collected_batches = []
        self.batch_count = 0
        self.pca_fitted = False

    def _initialize_as_identity(self):
        init.eye_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, x):
        batch, time = x.shape[0], x.shape[1]
        x_flat = x.view(batch * time, -1)  # Shape: (B*T, C*2)

        # Apply linear transform
        transformed = self.linear(x_flat)
        transformed = transformed.view(batch, time, self.n_channels, 2)  # [B, T, C, 2]

        # Apply nonlinearity → this is the "activation" we may PCA
        nonlin_output = self.nlin(transformed)  # Shape: [B, T, C, 2]

        if not self.pca_fitted:
            # Collect data for PCA fitting
            orig_shape = nonlin_output.shape  # [B, T, C, 2]
            flat_output = nonlin_output.detach().cpu().numpy().reshape(-1, 2 * self.n_channels)
            self.collected_batches.append(flat_output)
            self.batch_count += 1

            if self.batch_count == self.n_pca_batches:
                all_data = np.concatenate(self.collected_batches, axis=0)
                self.pca = PCA(n_components=self.n_pca_components)
                self.pca.fit(all_data)
                self.pca_fitted = True
                self.collected_batches = None  # free memory

            # Use original output during collection phase
            transformed_output = nonlin_output

        else:
            orig_shape = nonlin_output.shape  # [B, T, C, 2]
            device = nonlin_output.device
            flat = nonlin_output.detach().cpu().numpy().reshape(-1, 2 * self.n_channels)

            # Optional: preserve std (as in your MLP)
            flat_std = flat.std()

            # PCA reduce + reconstruct
            reduced = self.pca.transform(flat)  # [N, n_comp]
            reconstructed_flat = self.pca.inverse_transform(reduced)  # [N, D]

            # Rescale to match original std (optional but recommended)
            # if reconstructed_flat.std() > 1e-8:
            #     reconstructed_flat = reconstructed_flat / reconstructed_flat.std() * flat_std

            # Back to tensor and original shape
            transformed_output = torch.from_numpy(reconstructed_flat).to(device).reshape(orig_shape)

        return transformed_output



class NlinCore(nn.Module):
    def __init__(self, n_channels, num_layers=3, nonlinearity="silu"):
        super().__init__()
        self.n_channels = n_channels
        layers = []
        for i in range(num_layers):
            
            if i  <= 1:
                c =32
            else:
                c = 7
            layers.append(SingleLayerPerceptronPCA(n_channels, nonlinearity, n_pca_components=c))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MultiChannelMLP(nn.Module):
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
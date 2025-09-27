import torch.nn as nn
import torch.nn.init as init
import torch

import numpy as np
from sklearn.decomposition import PCA

from backbones.common_modules import (
    TxaFilterEnsembleTorch,
    RxaFilterEnsembleTorch,
)


class SingleLayerPerceptron(nn.Module):
    def __init__(self, n_channels, nonlinearity="silu"):
        super().__init__()
        self.n_channels = n_channels

        # Linear layer: input and output are both 2 * n_channels
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
        }[nonlinearity]

    def _initialize_as_identity(self):
        init.eye_(self.linear.weight)
        # Optional: zero out the bias
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


        self.n_pca_components = 20
        self.n_pca_batches = 1

        # PCA-related stat
        self.pca = None
        self.collected_batches = []
        self.batch_count = 0
        self.pca_fitted = False

    def forward(self, x, h_0=None):
        filtered_x = self.txa_filter_layers(x)
        nonlin_output = self.nlin_layer(filtered_x)  # e.g., shape [B, C, H, W] or [B, T, D]

        if not self.pca_fitted:
            # Flatten to [B * ..., feature_dim]
            orig_shape = nonlin_output.shape
            batch_size = orig_shape[0]
            # Flatten all but the batch dimension → [B, -1]
            flat_output = nonlin_output.detach().cpu().numpy().reshape(-1, 32)
            self.collected_batches.append(flat_output)
            self.batch_count += 1

            if self.batch_count == self.n_pca_batches:
                # Concatenate along batch dimension: [N * B, feature_dim]
                all_data = np.concatenate(self.collected_batches, axis=0)
                self.pca = PCA(n_components=self.n_pca_components)
                self.pca.fit(all_data)  # ✅ Now 2D: [n_samples, n_features]
                self.pca_fitted = True
                self.collected_batches = None  # free memory

            # Pass through original (or you could use identity)
            transformed_output = nonlin_output

        else:
            orig_shape = nonlin_output.shape  # e.g., [B, C, H, W]
            device = nonlin_output.device

            # Flatten to [B, D]
            flat = nonlin_output.detach().cpu().numpy().reshape(-1, 32)

            flat_std = flat.std()

            # Reduce and reconstruct
            reduced = self.pca.transform(flat)                # [B, n_comp]
            reconstructed_flat = self.pca.inverse_transform(reduced)  # [B, D]

            reconstructed_flat = reconstructed_flat/reconstructed_flat.std()*flat_std
            # Reshape back to original spatial layout
            transformed_output = torch.from_numpy(reconstructed_flat).to(device).reshape(orig_shape)

        # Continue with rest of network
        filt_rxa = self.rxa_filter_layers(transformed_output)
        output = self.bn_output(filt_rxa)
        return output

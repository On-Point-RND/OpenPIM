import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SlidingWindowDataset(Dataset):
    """
    Sliding-window dataset: history of length L predicts the current sample.

    features: (num_windows, L, C, 2) or (num_windows, L, 2) for single-channel
    targets:  (num_windows, C, 2) or (num_windows, 2) for single-channel
    """

    def __init__(self, features, targets):
        if features.shape[-1] != 2:
            raise ValueError(f"Expected I/Q last dim, got features {features.shape}")
        if features.ndim == 3:
            pass  # (N, L, 2)
        elif features.ndim == 4:
            pass  # (N, L, C, 2)
        else:
            raise ValueError(
                f"Expected features (N, L, 2) or (N, L, C, 2), got {features.shape}"
            )

        if targets.shape[-1] != 2:
            raise ValueError(f"Expected I/Q last dim, got targets {targets.shape}")
        if targets.ndim == 2:
            pass  # (N, 2)
        elif targets.ndim == 3:
            pass  # (N, C, 2)
        else:
            raise ValueError(
                f"Expected targets (N, 2) or (N, C, 2), got {targets.shape}"
            )

        if features.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Feature/target sample mismatch: {features.shape[0]} vs {targets.shape[0]}"
            )

        self.features = torch.Tensor(features)
        self.targets = torch.Tensor(targets)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def make_sliding_windows(
    x: np.ndarray,
    y: np.ndarray,
    history_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window samples: history [t-L+1 ... t] -> target y[t].

    Input:
      x: (N, C, 2)
      y: (N, C, 2)
    Output:
      X_win: (N-L+1, L, C, 2)
      Y_cur: (N-L+1, C, 2)
    """
    if x.ndim != 3 or x.shape[-1] != 2:
        raise ValueError(f"x must have shape (N, C, 2), got {x.shape}")
    if y.ndim != 3 or y.shape[-1] != 2:
        raise ValueError(f"y must have shape (N, C, 2), got {y.shape}")
    if x.shape != y.shape:
        raise ValueError(f"x/y shape mismatch: {x.shape} vs {y.shape}")
    if history_len <= 0:
        raise ValueError(f"history_len must be positive, got {history_len}")

    n, n_channels = x.shape[0], x.shape[1]
    if n < history_len:
        raise ValueError(f"Not enough samples: N={n}, history_len={history_len}")

    n_windows = n - history_len + 1
    x_windows = np.empty((n_windows, history_len, n_channels, 2), dtype=x.dtype)
    y_current = np.empty((n_windows, n_channels, 2), dtype=y.dtype)

    for i in range(n_windows):
        j = i + history_len
        x_windows[i] = x[i:j]
        y_current[i] = y[j - 1]

    return x_windows, y_current


def _prepare_ftdnn_tensors(
    x_win: np.ndarray,
    y_cur: np.ndarray,
    channel_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """FTDNN: one RF channel -> (N, L, 2) and (N, 2)."""
    n_channels = x_win.shape[2]
    if channel_idx < 0 or channel_idx >= n_channels:
        raise ValueError(
            f"channel_idx={channel_idx} is out of range for C={n_channels}"
        )
    return x_win[:, :, channel_idx, :], y_cur[:, channel_idx, :]


def _window_split(
    x: np.ndarray,
    y: np.ndarray,
    seq_len: int,
    backbone_type: str,
    channel_idx: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    x_w, y_w = make_sliding_windows(x, y, seq_len)
    n_channels = x_w.shape[2]
    if backbone_type == "ftdnn":
        x_w, y_w = _prepare_ftdnn_tensors(x_w, y_w, channel_idx)
        n_channels = 1
    return x_w, y_w, n_channels


def build_sliding_loaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seq_len: int,
    backbone_type: str,
    channel_idx: int = 0,
) -> tuple[tuple[DataLoader, DataLoader, DataLoader], int]:
    """Build train/val/test loaders for sliding-window samples."""
    x_train_w, y_train_w, n_channels = _window_split(
        x_train, y_train, seq_len, backbone_type, channel_idx
    )
    x_val_w, y_val_w, _ = _window_split(
        x_val, y_val, seq_len, backbone_type, channel_idx
    )
    x_test_w, y_test_w, _ = _window_split(
        x_test, y_test, seq_len, backbone_type, channel_idx
    )

    train_set = SlidingWindowDataset(x_train_w, y_train_w)
    val_set = SlidingWindowDataset(x_val_w, y_val_w)
    test_set = SlidingWindowDataset(x_test_w, y_test_w)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return (train_loader, val_loader, test_loader), n_channels

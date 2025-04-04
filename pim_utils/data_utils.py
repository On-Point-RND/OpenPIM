import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from modules.data_collector import (
    InfiniteIQSegmentDataset,
    IQSegmentDataset,
    prepare_data_for_predict,
    prepare_data,
)


class ComplexScaler:
    def __init__(self, data, path_dir_save):
        self.scales = self.get_mean_var(data, path_dir_save)

    def get_mean_var(self, data, path_dir_save):
        # Vectorized computation for X
        x_train = data["X"][
            "Train"
        ]  # np.stack(data["X"]["Train"], axis=0)  # Stack all IDs into a 3D array
        means_X = x_train.mean(axis=0)  # Compute mean along the sample axis
        sd_X = x_train.std(axis=0, ddof=0)  # Compute population standard deviation

        # Vectorized computation for Y
        y_train = data["Y"]["Train"]  # np.stack(data["Y"]["Train"], axis=0)
        means_Y = y_train.mean(axis=0)
        sd_Y = y_train.std(axis=0, ddof=0)
        # Return results as dictionaries

        noise = data["N"]["Train"]  # np.stack(data["Y"]["Train"], axis=0)
        means_N = noise.mean(axis=0)
        sd_N = noise.std(axis=0, ddof=0)

        pd.DataFrame(
            {
                "Value": ["real", "imag"],
                "mean_X": [
                    means_X[0],
                    means_X[1],
                ],
                "mean_Y": [
                    means_Y[0],
                    means_Y[1],
                ],
                "sd_X": [
                    sd_X[0],
                    sd_X[1],
                ],
                "sd_Y": [
                    sd_Y[0],
                    sd_Y[1],
                ],
            }
        ).to_csv(path_dir_save + "/means_sd.csv", index=False)

        return {
            "means": {"X": means_X, "Y": means_Y, "N": means_N},
            "sd": {"X": sd_X, "Y": sd_Y, "N": sd_N},
        }

    def normalize(self, x, key="X"):
        for c in range(2):
            x[..., c] = (x[..., c] - self.scales["means"][key][c]) / self.scales["sd"][
                key
            ][c]

        return x

    def rescale(self, x, key="X"):
        for c in range(2):
            x[..., c] = (
                x[..., c].flatten() * self.scales["sd"][key][c]
                + self.scales["means"][key][c]
            )

        return x


def load_resources(
    dataset_path: str,
    dataset_name: str,
    filter_path: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    n_back: int,
    n_fwd: int,
    batch_size: int,
    batch_size_eval: int,
    path_dir_save: str,
):
    # Load dataset
    path = os.path.join(dataset_path, dataset_name, f"{dataset_name}.mat")
    data = prepare_data(path, filter_path, train_ratio, val_ratio, test_ratio)

    input_size = 1 + n_back + n_fwd
    n_channels = len(data["X"]["Train"])

    # Calculate normalization parameters
    СScaler = ComplexScaler(data, path_dir_save)

    # Apply normalization and slice data
    for data_part in ["Train", "Val", "Test"]:
        data["X"][data_part] = СScaler.normalize(data["X"][data_part], key="X")
        data["Y"][data_part] = СScaler.normalize(data["Y"][data_part], key="Y")
        data["N"][data_part] = data["N"][data_part][n_back:-n_fwd, :]

    train_set = InfiniteIQSegmentDataset(
        data["X"]["Train"],
        data["Y"]["Train"],
        n_back=n_back,
        n_fwd=n_fwd,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

    # Validation set/loader
    val_set = IQSegmentDataset(
        data["X"]["Val"], data["Y"]["Val"], n_back=n_back, n_fwd=n_fwd
    )
    val_loader = DataLoader(val_set, batch_size=batch_size_eval, shuffle=False)

    # Test set/loader
    test_set = IQSegmentDataset(
        data["X"]["Test"], data["Y"]["Test"], n_back=n_back, n_fwd=n_fwd
    )
    test_loader = DataLoader(test_set, batch_size=batch_size_eval, shuffle=False)

    return (
        (train_loader, val_loader, test_loader),
        input_size,
        n_channels,
        data["N"],
        data["filter"],
        СScaler,
        data["specs"],
    )


# NOTE: need to validate this script
def prepare_residuals(
    data,
    n_back,
    n_fwd,
    batch_size,
    batch_size_eval,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
):
    # Combine I/Q components into 2D arrays
    txa = np.row_stack((data["I_txa"], data["Q_txa"])).T
    rxa = np.row_stack((data["I_rxa_new"], data["Q_rxa_new"])).T
    initial_rxa = np.row_stack((data["I_rxa_old"], data["Q_rxa_old"])).T
    nfa = np.row_stack((data["I_noise"], data["Q_noise"])).T

    # Split data into train/val/test sets
    total_samples = txa.shape[0]
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    data_dict = {
        "X": {
            "Train": txa[:train_end, :],
            "Val": txa[train_end:val_end, :],
            "Test": txa[val_end:, :],
        },
        "Y": {
            "Train": rxa[:train_end, :],
            "Val": rxa[train_end:val_end, :],
            "Test": rxa[val_end:, :],
        },
        "Y_initial": {
            "Train": initial_rxa[:train_end, :],
            "Val": initial_rxa[train_end:val_end, :],
            "Test": initial_rxa[val_end:, :],
        },
        "N": {
            "Train": nfa[:train_end, :],
            "Val": nfa[train_end:val_end, :],
            "Test": nfa[val_end:, :],
        },
    }

    # Calculate normalization parameters
    means_X = np.mean(data_dict["X"]["Train"], axis=0)
    means_Y = np.mean(data_dict["Y"]["Train"], axis=0)
    sd_X = np.std(data_dict["X"]["Train"], axis=0)
    sd_Y = np.std(data_dict["Y"]["Train"], axis=0)

    # Apply normalization
    for data_type in ["Train", "Val", "Test"]:
        # Normalize X
        data_dict["X"][data_type][:, 0] = (
            data_dict["X"][data_type][:, 0] - means_X[0]
        ) / sd_X[0]
        data_dict["X"][data_type][:, 1] = (
            data_dict["X"][data_type][:, 1] - means_X[1]
        ) / sd_X[1]

        # Normalize Y
        data_dict["Y"][data_type][:, 0] = (
            data_dict["Y"][data_type][:, 0] - means_Y[0]
        ) / sd_Y[0]
        data_dict["Y"][data_type][:, 1] = (
            data_dict["Y"][data_type][:, 1] - means_Y[1]
        ) / sd_Y[1]

        # Trim N and Y_initial
        data_dict["N"][data_type] = data_dict["N"][data_type][n_back:-n_fwd, :]
        data_dict["Y_initial"][data_type] = data_dict["Y_initial"][data_type][
            n_back:-n_fwd, :
        ]

    # Create datasets
    train_set = InfiniteIQSegmentDataset(
        [data_dict["X"]["Train"]],
        data_dict["Y"]["Train"],
        n_back=n_back,
        n_fwd=n_fwd,
    )

    val_set = IQSegmentDataset(
        [data_dict["X"]["Val"]],
        data_dict["Y"]["Val"],
        n_back=n_back,
        n_fwd=n_fwd,
    )

    test_set = IQSegmentDataset(
        [data_dict["X"]["Test"]],
        data_dict["Y"]["Test"],
        n_back=n_back,
        n_fwd=n_fwd,
    )

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size_eval, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size_eval, shuffle=False)

    return (
        (train_loader, val_loader, test_loader),
        data_dict["N"],
        data_dict["Y_initial"],
        {"X": means_X, "Y": means_Y},
        {"X": sd_X, "Y": sd_Y},
    )


def load_for_pred(
    dataset_path: str,
    dataset_name: str,
    path_dir_save: str,
    n_back: int,
    n_fwd: int,
    batch_size_eval: int,
):
    # Construct file path
    path = os.path.join(dataset_path, dataset_name, f"{dataset_name}.mat")

    # Load data
    data = prepare_data_for_predict(path)

    # Calculate input size
    input_size = 1 + n_back + n_fwd
    n_channels = len(data["X"])

    total_means = []
    total_sd = []
    total_pred_data = []
    total_pred_loaders = []

    for ch in range(n_channels):
        # Load normalization parameters
        csv_path = os.path.join(path_dir_save, f"CH_{ch}", "means_sd.csv")
        mean_sds = pd.read_csv(csv_path)

        means = {"X": mean_sds["mean_X"].tolist(), "Y": mean_sds["mean_Y"].tolist()}
        sd = {"X": mean_sds["sd_X"].tolist(), "Y": mean_sds["sd_Y"].tolist()}

        # Prepare prediction data
        pred_X = data["X"][ch][n_back:-n_fwd, :].copy()
        pred_Y = data["Y"][ch][n_back:-n_fwd, :].copy()
        pred_noise = data["noise"][ch][n_back:-n_fwd, :].copy()

        pred_data = {"X": pred_X, "Y": pred_Y, "noise": pred_noise}

        # Apply normalization
        data["X"][ch][:, 0] = (data["X"][ch][:, 0] - means["X"][0]) / sd["X"][0]
        data["X"][ch][:, 1] = (data["X"][ch][:, 1] - means["X"][1]) / sd["X"][1]

        data["Y"][ch][:, 0] = (data["Y"][ch][:, 0] - means["Y"][0]) / sd["Y"][0]
        data["Y"][ch][:, 1] = (data["Y"][ch][:, 1] - means["Y"][1]) / sd["Y"][1]

        # Create dataset and dataloader
        pred_set = IQSegmentDataset(
            data["X"],
            data["Y"][ch],
            n_back=n_back,
            n_fwd=n_fwd,
        )

        pred_loader = DataLoader(pred_set, batch_size=batch_size_eval, shuffle=False)

        # Collect results
        total_means.append(means)
        total_sd.append(sd)
        total_pred_data.append(pred_data)
        total_pred_loaders.append(pred_loader)

    return (
        total_pred_data,
        total_pred_loaders,
        input_size,
        n_channels,
        total_means,
        total_sd,
        data["FC_TX"],
        data["FS"],
    )

import pandas as pd
import numpy as np
from modules.loggers import make_logger

logger = make_logger()


def to2Dreal(x):
    """
    Convert complex array of shape (..., M) to real array of shape (..., M, 2)
    Then transpose to (N, M, 2) format for neural network input
    """
    # Stack real and imaginary parts along a new last axis
    real_imag = np.stack([x.real, x.imag], axis=-1)
    # Move the original first axis (M) to the second position, and N to the first
    return real_imag.transpose(1, 0, 2)


def toComplex(x):
    return x[..., 0] + 1j * x[..., 1]


# INFO: This is used in modules.datasets
def back_fwd_feature_prepare(sequence_x, sequence_t, n_back, n_fwd):
    win_len = n_back + n_fwd + 1
    num_samples = sequence_x.shape[0] - win_len + 1
    segments_x = np.zeros((num_samples, win_len, sequence_x.shape[1], 2), dtype=float)
    segments_y = np.zeros((num_samples, sequence_t.shape[1], 2), dtype=float)

    for step in range(num_samples):
        segments_x[step, :] = sequence_x[step : win_len + step, :]
        segments_y[step, :] = sequence_t[win_len + step - n_fwd - 1, :]

    return segments_x, segments_y


# INFO: This is used in modules.data_collector
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
                    means_X[..., 0],
                    means_X[..., 1],
                ],
                "mean_Y": [
                    means_Y[..., 0],
                    means_Y[..., 1],
                ],
                "sd_X": [
                    sd_X[..., 0],
                    sd_X[..., 1],
                ],
                "sd_Y": [
                    sd_Y[..., 0],
                    sd_Y[..., 1],
                ],
            }
        ).to_csv(path_dir_save + "/means_sd.csv", index=False)

        logger.success(f"Scaler was initiated, scaler shapes: {means_X.shape}")
        return {
            "means": {"X": means_X, "Y": means_Y, "N": means_N},
            "sd": {"X": sd_X, "Y": sd_Y, "N": sd_N},
        }

    def normalize(self, x, key="X"):
        x = (x - self.scales["means"][key]) / self.scales["sd"][key]
        return x

    def rescale(self, x, key="X"):
        x = x * self.scales["sd"][key] + self.scales["means"][key]
        return x

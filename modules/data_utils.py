import pandas as pd
import numpy as np
import os
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
import numpy as np
import pandas as pd
import os


class ComplexScaler:
    def __init__(self, data, path_dir_save, scaler_type="magnitude"):
        """
        Args:
            data (dict): Data dictionary containing 'X', 'Y', 'N' with 'Train' splits
            path_dir_save (str): Directory to save scaler parameters
            scaler_type (str): 'power' for power normalization, 'magnitude' for magnitude-based
        """
        self.scaler_type = scaler_type
        self.epsilon = 1e-8  # Prevent division by zero
        self.scales = self.get_scalers(data, path_dir_save)

    def get_scalers(self, data, path_dir_save):
        x_train = np.stack(data["X"]["train"], axis=0)
        y_train = np.stack(data["Y"]["train"], axis=0)
        noise = np.stack(data["N"]["train"], axis=0)

        if self.scaler_type == "power":
            scaling_x = self._compute_power_scale(x_train)
            scaling_y = self._compute_power_scale(y_train)
            scaling_n = self._compute_power_scale(noise)
            value_name = "power_scale"

        elif self.scaler_type == "magnitude":
            scaling_x = self._compute_magnitude_scale(x_train)
            scaling_y = self._compute_magnitude_scale(y_train)
            scaling_n = self._compute_magnitude_scale(noise)
            value_name = "magnitude_scale"

        else:
            raise ValueError("Unsupported scaler type. Use 'power' or 'magnitude'.")

        # Save to CSV
        df = pd.DataFrame(
            {
                "Value": [value_name],
                "X_scale": [scaling_x],
                "Y_scale": [scaling_y],
                "N_scale": [scaling_n],
            }
        )
        df.to_csv(os.path.join(path_dir_save, "scalers.csv"), index=False)
        return {"X": scaling_x, "Y": scaling_y, "N": scaling_n}

    def _compute_power_scale(self, data):
        """Compute scaling factor for power normalization (1/sqrt(avg_power))"""
        power = np.square(data[..., 0]) + np.square(data[..., 1])
        avg_power = np.mean(power)
        return np.sqrt(avg_power) + self.epsilon

    def _compute_magnitude_scale(self, data):
        """Compute scaling factor for magnitude-based normalization (1/max_magnitude)"""
        magnitudes = np.sqrt(np.square(data[..., 0]) + np.square(data[..., 1]))
        max_magnitude = np.max(magnitudes)
        return max_magnitude + self.epsilon

    def normalize(self, x, key="X"):
        """Normalize using the selected scaler type"""
        return x / self.scales[key]

    def rescale(self, x, key="X"):
        """Rescale to original scale"""
        return x * self.scales[key]

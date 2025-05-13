import os
import numpy as np
from scipy.io import loadmat
from torch.utils.data import DataLoader
from modules.datasets import InfiniteIQSegmentDataset, IQSegmentDataset
from modules.data_utils import ComplexScaler, to2Dreal
from modules.loggers import make_logger

logger = make_logger()


# INFO: This is the main script to load resources used in RUNNER, it runs at the begginign of training once
def load_resources(
    dataset_path: str,
    dataset_name: str,
    filter_path: str,
    PIM_type: str,
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
    data = load_and_split_data(
        path, filter_path, train_ratio, val_ratio, test_ratio, PIM_type
    )
    input_size = 1 + n_back + n_fwd
    n_channels = data["X"]["Train"].shape[1]

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

    logger.success(f"Dataloaders were created")
    return (
        (train_loader, val_loader, test_loader),
        input_size,
        n_channels,
        data["N"],
        data["filter"],
        СScaler,
        data["specs"],
    )


# INFO: This is used in the previous script, mainly to split the data
def load_and_split_data(
    data_path,
    filter_path,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    PIM_type="total",
):

    fil = loadmat(filter_path)["flt_coeff"]
    data = loadmat(data_path)

    print("Concidered PIM_type: ", PIM_type)

    if PIM_type == "cond":
        int_pim = data["PIM_COND"]
        rxa = to2Dreal(data["nfa"] + int_pim)
    elif PIM_type == "leak":
        try:
            int_pim = data["PIM_COND_LEAK"]
            rxa = to2Dreal(data["nfa"] + int_pim)
        except:
            raise ValueError(f"PIM type '{PIM_type}' is not supported.")
    elif PIM_type == "ext":
        rxa = to2Dreal(data["nfa"] + data["PIM_EXT"])
    elif PIM_type == "total":
        rxa = to2Dreal(data["rxa"])
    else:
        raise ValueError(f"PIM type '{PIM_type}' is not supported.")

    txa = to2Dreal(data["txa"])
    nfa = to2Dreal(data["nfa"])

    FC_TX = data["BANDS_DL"][0][0][0][0][0] / 10**6
    FC_RX = data["BANDS_UL"][0][0][0][0][0] / 10**6
    FS = data["Fs"][0][0] / 10**6
    PIM_SFT = data["PIM_sft"][0][0] / 10**6
    PIM_BW = data["BANDS_TX"][0][0][1][0][0] / 10**6
    PIM_total_BW = data["BANDS_TX"][0][0][3][0][0] / 10**6

    spec_dictionary = {
        "FC_TX": FC_TX,
        "FC_RX": FC_RX,
        "FS": FS,
        "PIM_SFT": PIM_SFT,
        "PIM_BW": PIM_BW,
        "PIM_total_BW": PIM_total_BW,
        "nperseg": 1536,
    }

    total_samples = txa.shape[0]
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    logger.success(f"Data load and split is done")
    return {
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
        "N": {
            "Train": nfa[:train_end, :],
            "Val": nfa[train_end:val_end, :],
            "Test": nfa[val_end:, :],
        },
        "specs": spec_dictionary,
        "filter": fil,
    }


if __name__ == "__main__":

    logger = make_logger()

    # INFO: run tests to reproduce
    dataset_path = "/home/dev/public-datasets/e.shvetsov/PIM/FOR_COOPERATION/"
    dataset_name = "16TR_C25Nc16CD_CL_E20Ne1CD_20250117_1L"  # "1TR_C20Nc1CD_E20Ne1CD_20250117_0.5m"
    filter_path = (
        "/home/dev/public-datasets/e.shvetsov/PIM/FOR_COOPERATION/rx_filter.mat"
    )
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2
    n_back = 128
    n_fwd = 1
    batch_size = 32
    batch_size_eval = 256
    path_dir_save = "./results"

    # INFO: validate complex scaler for 1 and 16 TR
    # INFO: size for scaler C x 2 (channels x 2)
    # INFO: size for the data L x C x 2 (length, channels, 2)

    path = os.path.join(dataset_path, dataset_name, f"{dataset_name}.mat")
    data = load_and_split_data(path, filter_path, train_ratio, val_ratio, test_ratio)
    CScaler = ComplexScaler(data, dataset_path)
    logger.info(f"X shape:  {data['X']['Train'].shape}")
    logger.info(f"X Scales shape: {CScaler.scales['means']['X'].shape}")
    normalized = CScaler.normalize(data["X"]["Train"], key="X")
    rescaled = CScaler.rescale(normalized, key="X")
    assert data["X"]["Train"].shape == rescaled.shape
    assert np.allclose(data["X"]["Train"], rescaled, atol=1e-6) is True

    # INFO: validate datasets and backward splits

    (
        (train_loader, val_loader, test_loader),
        input_size,
        n_channels,
        noise,
        filter,
        CScaler,
        specs,
    ) = load_resources(
        dataset_path,
        dataset_name,
        filter_path,
        train_ratio,
        val_ratio,
        test_ratio,
        n_back,
        n_fwd,
        batch_size,
        batch_size_eval,
        path_dir_save,
    )


# def data_prepare(X, y, frame_length, degree):
#     Input = []
#     Output = []
#     X = torch.Tensor(X)
#     y = torch.Tensor(y)
#     for k in range(X.shape[0]):
#         Complex_In = torch.complex(X[k, :, 0], X[k, :, 1])
#         Complex_Out = torch.complex(y[k, :, 0], y[k, :, 1])
#         ulength = len(Complex_In) - frame_length
#         Input_matrix = torch.complex(
#             torch.zeros(ulength, frame_length), torch.zeros(ulength, frame_length)
#         )
#         degree_matrix = torch.complex(
#             torch.zeros(ulength - frame_length, frame_length * frame_length * degree),
#             torch.zeros(ulength - frame_length, frame_length * frame_length * degree),
#         )
#         for i in range(ulength):
#             Input_matrix[i, :] = Complex_In[i : i + frame_length]
#         for j in range(1, degree):
#             for h in range(frame_length):
#                 degree_matrix[
#                     :,
#                     (j - 1) * frame_length * frame_length
#                     + h * frame_length : (j - 1) * frame_length * frame_length
#                     + (h + 1) * frame_length,
#                 ] = Input_matrix[: ulength - frame_length] * torch.pow(
#                     abs(Input_matrix[h : h + ulength - frame_length, :]), j
#                 )
#         Input_matrix = torch.cat(
#             (Input_matrix[: ulength - frame_length], degree_matrix), dim=1
#         )
#         b_output = np.array(Complex_Out[: len(Complex_In) - 2 * frame_length])
#         b_input = np.array(Input_matrix)
#         Input.append(b_input)
#         Output.append(b_output)

#     return Input, Output


# def load_for_pred(
#     dataset_path: str,
#     dataset_name: str,
#     path_dir_save: str,
#     n_back: int,
#     n_fwd: int,
#     batch_size_eval: int,
# ):
#     # Construct file path
#     path = os.path.join(dataset_path, dataset_name, f"{dataset_name}.mat")

#     # Load data
#     data = prepare_data_for_predict(path)

#     # Calculate input size
#     input_size = 1 + n_back + n_fwd
#     n_channels = len(data["X"])

#     total_means = []
#     total_sd = []
#     total_pred_data = []
#     total_pred_loaders = []

#     for ch in range(n_channels):
#         # Load normalization parameters
#         csv_path = os.path.join(path_dir_save, f"CH_{ch}", "means_sd.csv")
#         mean_sds = pd.read_csv(csv_path)

#         means = {"X": mean_sds["mean_X"].tolist(), "Y": mean_sds["mean_Y"].tolist()}
#         sd = {"X": mean_sds["sd_X"].tolist(), "Y": mean_sds["sd_Y"].tolist()}

#         # Prepare prediction data
#         pred_X = data["X"][ch][n_back:-n_fwd, :].copy()
#         pred_Y = data["Y"][ch][n_back:-n_fwd, :].copy()
#         pred_noise = data["noise"][ch][n_back:-n_fwd, :].copy()

#         pred_data = {"X": pred_X, "Y": pred_Y, "noise": pred_noise}

#         # Apply normalization
#         data["X"][ch][:, 0] = (data["X"][ch][:, 0] - means["X"][0]) / sd["X"][0]
#         data["X"][ch][:, 1] = (data["X"][ch][:, 1] - means["X"][1]) / sd["X"][1]

#         data["Y"][ch][:, 0] = (data["Y"][ch][:, 0] - means["Y"][0]) / sd["Y"][0]
#         data["Y"][ch][:, 1] = (data["Y"][ch][:, 1] - means["Y"][1]) / sd["Y"][1]

#         # Create dataset and dataloader
#         pred_set = IQSegmentDataset(
#             data["X"],
#             data["Y"][ch],
#             n_back=n_back,
#             n_fwd=n_fwd,
#         )

#         pred_loader = DataLoader(pred_set, batch_size=batch_size_eval, shuffle=False)

#         # Collect results
#         total_means.append(means)
#         total_sd.append(sd)
#         total_pred_data.append(pred_data)
#         total_pred_loaders.append(pred_loader)

#     return (
#         total_pred_data,
#         total_pred_loaders,
#         input_size,
#         n_channels,
#         total_means,
#         total_sd,
#         data["FC_TX"],
#         data["FS"],
#     )

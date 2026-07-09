import os
import numpy as np
import torch
from scipy.io import loadmat
from modules.dataset_sequential import build_sequential_loaders
from modules.dataset_sliding import build_sliding_loaders
from modules.data_utils import ComplexScaler, to2Dreal
from modules.loggers import make_logger

logger = make_logger()

DATASET_MODES = ("sequential", "sliding")


def _ensure_n_c_2(x: np.ndarray, name: str) -> np.ndarray:
    """
    Ensure signal tensor shape is (N, C, 2).

    Accepts:
      - (N, C, 2): returned as is
      - (N, 2): interpreted as single-channel -> (N, 1, 2)
    """
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[-1] == 2:
        return x
    if x.ndim == 2 and x.shape[-1] == 2:
        return x[:, None, :]
    raise ValueError(
        f"{name} must have shape (N, C, 2) or (N, 2), got {x.shape}"
    )


def resolve_dataset_path(dataset_path: str, dataset_name: str) -> str:
    """Flat .pt, nested .pt subfolder, or legacy .mat subfolder."""
    flat_pt = os.path.join(dataset_path, f"{dataset_name}.pt")
    if os.path.isfile(flat_pt):
        return flat_pt
    nested_pt = os.path.join(dataset_path, dataset_name, f"{dataset_name}.pt")
    if os.path.isfile(nested_pt):
        return nested_pt
    mat_path = os.path.join(dataset_path, dataset_name, f"{dataset_name}.mat")
    if os.path.isfile(mat_path):
        return mat_path
    raise FileNotFoundError(
        f"Dataset not found: tried {flat_pt}, {nested_pt}, and {mat_path}"
    )


def _load_data_file(path: str) -> dict:
    if path.endswith(".pt"):
        raw = torch.load(path, map_location="cpu", weights_only=False)
        return {
            k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
            for k, v in raw.items()
        }
    return loadmat(path)


def load_rx_filter_coeff(filter_path: str) -> np.ndarray:
    """
    Load RX filter coefficients from ``.npy`` or ``.mat``.

    ``.mat`` fields supported:
      - ``flt_coeff``
      - ``rx_filter``
    """
    if filter_path.endswith(".npy"):
        return np.asarray(np.load(filter_path)).squeeze()

    rx_filter_keys = ("flt_coeff", "rx_filter")
    mat = loadmat(filter_path)
    for key in rx_filter_keys:
        if key in mat:
            return np.asarray(mat[key]).squeeze()

    available = [k for k in mat if not k.startswith("__")]
    raise KeyError(
        f"No filter field in {filter_path}. "
        f"Tried {rx_filter_keys}, found {available}"
    )


def _is_pt_data(data: dict) -> bool:
    return "PIM_fc" in data and "BANDS_DL" not in data


def _specs_from_data(data: dict, data_type: str) -> dict:
    if _is_pt_data(data):
        return {
            "FC_TX": float(data["Fc"]) / 1e6,
            "FC_RX": float(data["Fc"] + data["CS"]) / 1e6,
            "FS": float(data["Fs"]) / 1e6,
            "PIM_SFT": float(data["PIM_fc"]) / 1e6,
            "PIM_BW": float(data["PIM_bw"]) / 1e6,
            "PIM_total_BW": float(data["CS"]) / 1e6,
            "nperseg": 1536,
        }

    if data_type == "synth":
        return {
            "FC_TX": data["BANDS_DL"][0][0][0][0][0] / 10**6,
            "FC_RX": data["BANDS_UL"][0][0][0][0][0] / 10**6,
            "FS": data["Fs"][0][0] / 10**6,
            "PIM_SFT": data["PIM_sft"][0][0] / 10**6,
            "PIM_BW": data["BANDS_TX"][0][0][1][0][0] / 10**6,
            "PIM_total_BW": data["BANDS_TX"][0][0][3][0][0] / 10**6,
            "nperseg": 1536,
        }

    return {
        "FC_TX": 1842.5,
        "FC_RX": 0,
        "FS": 245.76,
        "PIM_SFT": 15,
        "PIM_BW": 5,
        "PIM_total_BW": 30,
        "nperseg": 1536,
    }


# INFO: This is the main script to load resources used in RUNNER,
# it runs at the begginign of training once
def load_resources(
    dataset_path: str,
    dataset_name: str,
    filter_path: str,
    PIM_type: str,
    data_type: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    batch_size: int,
    batch_size_eval: int,
    seq_len: int,
    path_dir_save: str,
    backbone_type: str = "mcp",
    dataset_mode: str = "sequential",
    channel_idx: int = 0,
):
    # Load dataset (.pt flat file or legacy .mat subfolder)
    path = resolve_dataset_path(dataset_path, dataset_name)
    data = load_and_split_data(
        path, filter_path,
        train_ratio, val_ratio, test_ratio,
        PIM_type, data_type
    )
    # Support both multi-channel (N, C, 2) and already single-channel (N, 2) inputs.
    for part in ["train", "val", "test"]:
        data["X"][part] = _ensure_n_c_2(data["X"][part], f"X/{part}")
        data["Y"][part] = _ensure_n_c_2(data["Y"][part], f"Y/{part}")
        data["N"][part] = _ensure_n_c_2(data["N"][part], f"N/{part}")

    # Calculate normalization parameters
    СScaler = ComplexScaler(data, path_dir_save)
    
    # Apply normalization and slice data
    for data_part in ["train", "val", "test"]:
        data["X"][data_part] = СScaler.normalize(data["X"][data_part], key="X")
        data["Y"][data_part] = СScaler.normalize(data["Y"][data_part], key="Y")
    
    if dataset_mode not in DATASET_MODES:
        raise ValueError(
            f"dataset_mode must be one of {DATASET_MODES}, got '{dataset_mode}'"
        )
    logger.info(f"Dataset mode: {dataset_mode}")

    if dataset_mode == "sliding":
        (train_loader, val_loader, test_loader), n_channels = build_sliding_loaders(
            data["X"]["train"],
            data["Y"]["train"],
            data["X"]["val"],
            data["Y"]["val"],
            data["X"]["test"],
            data["Y"]["test"],
            seq_len=seq_len,
            backbone_type=backbone_type,
            channel_idx=channel_idx,
        )
    else:
        (train_loader, val_loader, test_loader), n_channels = build_sequential_loaders(
            data["X"]["train"],
            data["Y"]["train"],
            data["X"]["val"],
            data["Y"]["val"],
            data["X"]["test"],
            data["Y"]["test"],
            seq_len=seq_len,
            batch_size=batch_size,
            batch_size_eval=batch_size_eval,
        )

    logger.success(f"Dataloaders were created")
    return (
        (train_loader, val_loader, test_loader),
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
    data_type = 'synth'
):

    fil = load_rx_filter_coeff(filter_path)
    data = _load_data_file(data_path)

    print("Considered PIM type: ", PIM_type)

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

    spec_dictionary = _specs_from_data(data, data_type)

    total_samples = txa.shape[0]
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    logger.success(f"Data load and split is done")
    return {
        "X": {
            "train": txa[:train_end, :],
            "val": txa[train_end:val_end, :],
            "test": txa[val_end:, :],
        },
        "Y": {
            "train": rxa[:train_end, :],
            "val": rxa[train_end:val_end, :],
            "test": rxa[val_end:, :],
        },
        "N": {
            "train": nfa[:train_end, :],
            "val": nfa[train_end:val_end, :],
            "test": nfa[val_end:, :],
        },
        "specs": spec_dictionary,
        "filter": fil,
    }


if __name__ == "__main__":

    logger = make_logger()

    # INFO: run tests to reproduce
    dataset_path = "/home/dev/public-datasets/e.shvetsov/PIM/FOR_COOPERATION/"
    dataset_name = "16TR_C25Nc16CD_CL_E20Ne1CD_20250117_1L"
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
    logger.info(f"X shape:  {data['X']['train'].shape}")
    logger.info(f"X Scales shape: {CScaler.scales['means']['X'].shape}")
    normalized = CScaler.normalize(data["X"]["train"], key="X")
    rescaled = CScaler.rescale(normalized, key="X")
    assert data["X"]["train"].shape == rescaled.shape
    assert np.allclose(data["X"]["train"], rescaled, atol=1e-6) is True

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
